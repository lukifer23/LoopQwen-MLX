"""Train a recurrent vocabulary exit on native, ordinary-depth reasoning prefixes."""

import argparse
import gc
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim

from .. import REVISION
from ..deployment import load_trained
from ..evaluation.language7 import base
from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import checkpoint_location, digest, read, save
from . import bootstrap, core, deliberation, paired_solutions, rationale_transfer

NAME = "learned-exit-01"
REG = Path("research/experiments") / (NAME + ".json")
PANEL = Path("research/data") / (NAME + ".json")
TRAIN = PANEL.with_name("learned-exit-training.json")


def material(s):
    exclude = [row for f in s["excluded_panels"] for row in read(f)]
    used = {i for f in s["excluded_training"] for i in read(f)["ordered_questions"]}
    _, rows = paired_solutions.material(s, exclude, exclude_ids=used)
    drafts = Path(s["drafts"])
    if digest(drafts) != s["draft_sha256"]:
        raise ValueError("Training draft digest differs")
    pool = read(deliberation.TRAIN)
    saved = unpack(drafts)
    if [r["id"] for r in saved] != [r["id"] for r in pool]:
        raise ValueError(
            "Training-only Base drafts differ from the frozen question order"
        )
    manifest = {"ordered_questions": [r["id"] for r in pool], "drafts": digest(drafts)}
    if set(manifest["ordered_questions"]) & {r["id"] for r in rows}:
        raise ValueError("Training question in evaluation")
    return pool, saved, rows, manifest


def extend(s, root, pool, originals, budget):
    m, t, _ = base()
    path = root / "extensions.json.gz"
    extensions = unpack(path) if path.exists() else []
    for i, (item, old) in enumerate(zip(pool, originals, strict=True)):
        if i < len(extensions):
            if extensions[i]["id"] != item["id"]:
                raise ValueError("Wrong saved rollout")
            continue
        suffix = []
        seconds = 0.0
        if old["close_at"] is None and old["stop"] == "cap":
            prefix = old["token_ids"]
            val = core.generate(
                m,
                t,
                item["prompt_ids"] + prefix,
                s["training_draft_cap"] - len(prefix),
                budget.check,
                thinking=True,
            )
            suffix, seconds = val["token_ids"], val["seconds"]
        extensions.append({"id": item["id"], "token_ids": suffix, "seconds": seconds})
        packed(path, extensions)
    return extensions


def examples(s, tokenizer, pool, originals, extensions):
    docs, traces = [], []
    close = core.atom(tokenizer, "</think>")
    stopping = {core.atom(tokenizer, "<|im_end|>"), *tokenizer.eos_token_ids}
    for row, original, ext in zip(pool, originals, extensions, strict=True):
        ids = original["token_ids"] + ext["token_ids"]
        until = next(
            (i for i, v in enumerate(ids) if v == close or v in stopping), len(ids)
        )
        p = row["prompt_ids"]
        suffix = tokenizer.encode(
            "</think>\n\nFinal answer: " + row["gold"], add_special_tokens=False
        ) + [core.atom(tokenizer, "<|im_end|>")]
        points = (min(until, s["switch_after"]), until)
        for k in points:
            docs.append({"ids": p + ids[:k] + suffix, "prefix": len(p) + k})
        neg = sorted(
            {
                k
                for k in (
                    until // 4,
                    until // 2,
                    min(until - 1, s["switch_after"] // 2),
                )
                if 0 <= k < min(points)
            }
        )
        traces.append((p, ids[:until], sorted(set(points)), neg))
    return docs, traces


def fit_readout(s, m, traces, close, budget, root):
    m.eval()
    features, margins, labels = [], [], []
    for prompt, response, pos, neg in traces:
        budget.check()
        points = sorted([*pos, *neg])
        cache = m.make_cache()
        tokens = prompt + response
        for start in range(0, len(tokens), 128):
            chunk = tokens[start : start + 128]
            h = m.hidden(mx.array([chunk]), cache)
            mx.eval(h)
            for k in points:
                i = len(prompt) + k - 1 - start
                if 0 <= i < len(chunk):
                    state = h[0, i].astype(mx.float32)
                    logits = m.project(h[:, i : i + 1]).reshape(-1).astype(mx.float32)
                    # log-odds of the original close against the rest of the vocabulary.
                    odds = logits[close] - mx.logaddexp(
                        mx.logsumexp(logits[:close]), mx.logsumexp(logits[close + 1 :])
                    )
                    features.append(
                        mx.concatenate([state, mx.array([k / s["switch_after"], 1.0])])
                    )
                    margins.append(odds)
                    labels.append(float(k in pos))
                    mx.eval(features[-1], margins[-1])
    x, offsets, y = mx.stack(features), mx.stack(margins), mx.array(labels)
    weights = mx.zeros((x.shape[1],))
    importance = mx.where(y, 0.5 / y.mean(), 0.5 / (1 - y.mean()))

    def loss(w):
        z = offsets + x @ w
        return (importance * (mx.logaddexp(0.0, z) - y * z)).mean() + s["head_l2"] * (
            w * w
        ).sum()

    vg = mx.value_and_grad(loss)
    optimizer = optim.Adam(s["head_lr"])
    first = loss(weights).item()
    for _ in range(s["head_updates"]):
        budget.check()
        _, grad = vg(weights)
        weights = optimizer.apply_gradients({"w": grad}, {"w": weights})["w"]
        mx.eval(weights, optimizer.state)
    save(
        root / "readout-fit.json",
        {
            "positives": sum(labels),
            "negatives": len(labels) - sum(labels),
            "before": first,
            "after": loss(weights).item(),
            "features": x.shape[1],
            "updates": s["head_updates"],
        },
    )
    mx.save_safetensors(
        str(root / "model" / "exit-readout.safetensors"), {"weight": weights}
    )


def run(freeze):
    s = read(REG)
    pool, originals, rows, manifest = material(s)
    for p, val in ((PANEL, rows), (TRAIN, manifest)):
        if freeze and not p.exists():
            save(p, val)
        if read(p) != val:
            raise ValueError("Frozen data differ")
    if freeze:
        return
    reg = read("research/registry.json")
    if reg["experiments"][NAME]["registration"] != digest(REG) or any(
        reg["datasets"][str(p)]["sha256"] != digest(p) for p in (PANEL, TRAIN)
    ):
        raise ValueError("Unregistered snapshot")
    for k, key in (("parent", "parent_sha256"), ("dense_control", "dense_sha256")):
        if digest(Path(s[k]) / "adapters.safetensors") != s[key]:
            raise ValueError("Control identity changed")
    root = core.OUT / NAME
    root.mkdir(parents=True, exist_ok=True)
    if (root / "decision.json").exists():
        raise FileExistsError(root)
    b = core.Budget(NAME, s["limits"])
    result = {
        "registration": digest(REG),
        "data": digest(PANEL),
        "training": digest(TRAIN),
        "passed": False,
    }
    try:
        extensions = extend(s, root, pool, originals, b)
        gc.collect()
        mx.clear_cache()
        m, t, state = load_trained(s["parent"])
        docs, traces = examples(s, t, pool, originals, extensions)
        m.checkpointing = True
        dep = state["spec"]["deployment"]
        cp = Path("checkpoints/research") / NAME
        st = core.train(m, docs, dict(s, deployment=dep), cp, b, s["parent_sha256"])
        gen = checkpoint_location(cp)
        bootstrap.release(gen, root / "model")
        w = digest(gen / "adapters.safetensors")
        fit_readout(s, m, traces, core.atom(t, "</think>"), b, root)
        readout = digest(root / "model" / "exit-readout.safetensors")
        save(
            root / "model" / "deliberation.json",
            {
                "version": 2,
                "revision": REVISION,
                "switch_after": s["switch_after"],
                "adapter_sha256": w,
                "readout_sha256": readout,
            },
        )
        result.update(
            weights=w,
            readout=readout,
            updates=st["step"],
            targets=sum(v["targets"] for v in st["history"]),
            documents=fingerprint(docs),
        )
        del m, t
        gc.collect()
        mx.clear_cache()
        rationale_transfer.evaluate_pair(
            root,
            root / "model",
            rows,
            s,
            w,
            b,
            result,
            retention=True,
            decoder=deliberation.decode,
        )
        full = result["full"]
        result["passed"] = (
            full["all"]["wins"] - full["all"]["losses"] >= s["net_floor"]
            and all(full[k]["wins"] > full[k]["losses"] for k in s["sources"])
            and full["all"]["candidate"]["seconds"] <= full["all"]["base"]["seconds"]
            and result["retention"]["candidate"]["nll"]
            <= result["retention"]["base"]["nll"] + 0.03
        )
    except (MemoryError, TimeoutError, InterruptedError) as exc:
        result.update(status="interrupted", reason=str(exc))
    finally:
        b.close()
    save(root / "decision.json", result)
    print(result["status"], result["passed"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    run(parser.parse_args().freeze)
