"""Attribute three-fact induction directly to repeated execution of Qwen layer 15."""

import argparse
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten

from .. import REVISION
from ..evaluation.language7 import base
from ..evaluation.mechanism import packed
from ..evaluation.statistics import exact_mcnemar
from ..evidence_loop import EvidenceView, passage
from ..io import digest, read, save
from ..native_evidence import AnchoredNativeEvidence, NativeEvidence, prefix_regions
from . import core, deduction, induction_data

PANEL = Path("research/data/native-induction-01.json")


def fit(model, t, rows, depth, s, b, out):
    vocab, (features, _), arrays = deduction.training_inputs(
        model, t, rows, depth, b, prefix_regions, response=induction_data.response
    )
    mx.random.seed(s["fit_seed"])
    cls = (
        AnchoredNativeEvidence
        if s.get("architecture") == "anchored_native"
        else NativeEvidence
    )
    net = cls(model, s["rank"], vocab)
    net.center = features.mean(0)
    net.scale = mx.rsqrt(mx.var(features, axis=0) + 0.0001)
    columns = mx.array(
        [vocab.index(core.atom(t, " " + w)) for w in induction_data.WORDS]
    )
    lr = s["lr"]
    if "decay_start" in s:
        lr = optim.join_schedules(
            [
                optim.linear_schedule(0.1 * lr, lr, 50),
                optim.linear_schedule(lr, lr, 1),
                optim.cosine_decay(lr, s["updates"] - s["decay_start"], end=0.1 * lr),
            ],
            [50, s["decay_start"]],
        )
    optimizer = optim.Adam(lr)

    def loss(net, f, q, h, z, rest, y, support):
        queries, facts = net.states(f, q, depth)
        memory = net.readout(queries[-1])
        logits = z + net.bias(h, memory)
        nll = (
            mx.logaddexp(mx.logsumexp(logits, axis=-1), rest)
            - mx.take_along_axis(logits, y[..., None], axis=-1)[..., 0]
        )
        weight = 1 + 5 * mx.any(y[..., None] == columns, axis=-1)
        total = mx.sum(weight * nll) / mx.sum(weight) + s["auxiliary"] * net.auxiliary(
            queries, facts, support
        )
        if s.get("isolated_answer_weight"):
            labels = mx.any(y[..., None] == columns, axis=1).argmax(axis=-1)
            total = total + s["isolated_answer_weight"] * nn.losses.cross_entropy(
                net.content(memory)[:, columns], labels, reduction="mean"
            )
        return total

    vg = nn.value_and_grad(net, loss)
    rng = random.Random(s["fit_seed"])
    logs = []
    f, _, q, *rest = arrays
    start = time.perf_counter()
    for i in range(s["updates"]):
        b.check()
        ix = mx.array(rng.sample(range(len(rows)), s["batch"]))
        val, g = vg(net, *[v[ix] for v in (f, q, *rest)])
        g, _ = optim.clip_grad_norm(g, 1.0)
        optimizer.update(net, g)
        mx.eval(val, net.trainable_parameters(), optimizer.state)
        if i % 50 == 0 or i + 1 == s["updates"]:
            logs.append([i + 1, val.item()])
            print("native induction fit", i + 1, round(val.item(), 4), flush=True)
    path = out / "reader.safetensors"
    weights = dict(tree_flatten(net.trainable_parameters()))
    deltas = {
        name: mx.linalg.norm(v.astype(mx.float32)).item()
        for name, v in weights.items()
        if name.startswith("layer.") and name.endswith("lora_b")
    }
    if len(deltas) != 4 or not all(0 < v < float("inf") for v in deltas.values()):
        raise ValueError("Missing learned update to the native attention span")
    mx.save_safetensors(str(path), dict(**weights, center=net.center, scale=net.scale))
    save(
        out / "reader.json",
        {
            "depth": depth,
            "architecture": s.get("architecture", "native"),
            "base": REVISION,
            "rank": s["rank"],
            "vocab": vocab,
            "updates": s["updates"],
            "fit_seed": s["fit_seed"],
            "weights": digest(path),
            "loss": logs,
            "optimization_seconds": time.perf_counter() - start,
            "trainable_parameters": sum(x.size for x in weights.values()),
            "native_update_norms": deltas,
        },
    )
    return net


def restore(model, folder, expected):
    spec = read(folder / "reader.json")
    path = folder / "reader.safetensors"
    if (
        digest(path) != spec["weights"]
        or spec["weights"] != expected
        or spec["base"] != REVISION
    ):
        raise ValueError("Wrong native reader lineage")
    cls = (
        AnchoredNativeEvidence
        if spec.get("architecture") == "anchored_native"
        else NativeEvidence
    )
    net = cls(model, spec["rank"], spec["vocab"])
    weights = mx.load(str(path))
    if set(weights) != set(dict(tree_flatten(net.trainable_parameters()))) | {
        "center",
        "scale",
    }:
        raise ValueError("Incomplete learned native tensors")
    net.load_weights(list(weights.items()), strict=False)
    return net


def run(name, freeze):
    reg = Path("research/experiments") / (name + ".json")
    s = read(reg)
    panel = Path(s.get("material", PANEL))
    if freeze:
        data = induction_data.material(s)
        if panel.exists() and read(panel) != data:
            raise ValueError("Induction material changed")
        if not panel.exists():
            save(panel, data)
        return
    idx = read("research/registry.json")
    if idx["datasets"][str(panel)]["sha256"] != digest(panel) or idx["experiments"][
        name
    ]["registration"] != digest(reg):
        raise ValueError("Unregistered induction comparison")
    if s.get("requires"):
        arc = Path("research/results") / s["requires"]
        if (
            digest(arc / "checksums.json") != s["prerequisite_sha256"]
            or not read(arc / "decision.json")["passed"]
        ):
            raise ValueError("Failed induction prerequisite")
    out = core.OUT / name
    out.mkdir(exist_ok=False, parents=True)
    b = core.Budget(name, s["limits"])
    result = {"data": digest(panel), "registration": digest(reg), "passed": False}
    try:
        data = read(panel)
        rows = data[s["panel"]]
        model, t, _ = base()
        words = [core.atom(t, " " + w) for w in induction_data.WORDS]
        common = t.encode(
            "After considering the evidence.\n</think>\n\nThe answer is",
            add_special_tokens=False,
        )
        reports = {}
        records = {}
        for arm, c in s["arms"].items():
            depth = c["depth"]
            if c.get("fit"):
                net = fit(model, t, data["train"], depth, s, b, out)
            elif depth:
                net = restore(model, Path(c["reader"]), c["weights"])
            samples = []
            for r in rows:
                b.check()
                start = time.perf_counter()
                view = model
                if depth:
                    parts = passage(t, r["prompt_ids"])
                    if parts != (r["facts"], r["question"]):
                        raise ValueError("Different story tokens")
                    f, _ = prefix_regions(model, t, [*parts[0], parts[1]], 1)
                    q, stage_inputs = net.states(f[None, :-1], f[None, -1], depth)
                    view = EvidenceView(model, net, net.readout(q[-1]))
                v = core.generate(
                    view,
                    t,
                    r["prompt_ids"],
                    s["generation_cap"],
                    b.check,
                    thinking=True,
                )
                total = time.perf_counter() - start
                logits = view.project(
                    view.hidden(mx.array([r["prompt_ids"] + common]))[:, -1]
                )[0, mx.array(words)].tolist()
                heads = {}
                if depth and isinstance(net, AnchoredNativeEvidence):
                    support_scores = [net.native_scores(h) for h in stage_inputs]
                    heads["support_probabilities"] = [
                        mx.softmax(
                            support_scores[min(i, depth - 1)][:, 2 * i : 2 * i + 2],
                            axis=-1,
                        )
                        .mean(axis=1)[0]
                        .tolist()
                        for i in range(3)
                    ]
                ans = (
                    induction_data.answer(v["visible"], r["question"])
                    if v["complete"]
                    else None
                )
                samples.append(
                    dict(
                        **v,
                        id=r["id"],
                        answer=ans,
                        correct=ans == r["answer"],
                        probe=induction_data.WORDS[
                            max(range(4), key=logits.__getitem__)
                        ],
                        probe_logits=logits,
                        end_to_end=total,
                        **heads,
                    )
                )
                packed(out / (arm + ".json.gz"), {"records": samples})
            reports[arm] = deduction.summary(rows, samples)
            records[arm] = samples
        a, c = reports.values()
        delta = [
            int(x["correct"]) - int(y["correct"])
            for x, y in zip(*records.values(), strict=True)
        ]
        wins, losses = delta.count(1), delta.count(-1)
        p = exact_mcnemar(wins, losses)
        result.update(
            status="completed",
            scores=reports,
            wins=wins,
            losses=losses,
            p=p,
            passed=wins - losses >= s["net_floor"]
            and p < 0.05
            and a["probe"] >= c["probe"] + s["probe_floor"]
            and a["seconds"] <= s["time_ratio"] * c["seconds"],
        )
    except (MemoryError, TimeoutError, InterruptedError) as e:
        result.update(status="interrupted", reason=str(e))
    finally:
        b.close()
    save(out / "decision.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    run(args.name, args.freeze)
