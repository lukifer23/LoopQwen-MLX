"""Train only a differentiable passage relevance prior and test native full-context attention."""

import argparse
import gc
import shutil
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten, tree_map
from mlx_lm.utils import load_tokenizer

from .. import checkpoint
from ..attention_memory import AttentionReader, PassageScorer
from ..deployment import load_trained
from ..evaluation.language7 import base
from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import checkpoint_location, digest, read, save
from ..model_loader import local_snapshot
from . import bootstrap, core, ledger_curriculum, memory_data, rationale

REG = Path("research/experiments/attention-memory-01.json")
DATA = Path("research/data/attention-memory-01.json")


def fit(s, model, scorer, items, b, out):
    def loss(net, embedded, at):
        scores = 40 * net(embedded)
        centers = mx.arange(scores.size) * net.stride + net.span_width // 2
        positive = mx.abs(centers - (at - 2)) <= 4
        return mx.logsumexp(scores) - mx.logsumexp(
            mx.where(positive, scores, -float("inf"))
        )

    value_grad = nn.value_and_grad(scorer, loss)
    opt = optim.Adam(learning_rate=s["lr"])
    cp = Path("checkpoints/research") / s["id"]
    cp.mkdir(parents=True, exist_ok=True)
    identity = fingerprint(
        [
            digest(REG),
            digest(DATA),
            s["parent_sha256"],
            digest(Path(__file__).with_name("memory_data.py")),
            digest(Path(__file__).parents[1] / "attention_memory.py"),
        ]
    )
    state = {"step": 0, "identity": identity, "history": []}
    if (cp / "latest.json").exists():
        state = checkpoint.restore(cp, scorer, opt)
        if (
            state["identity"] != identity
            or int(opt.state["step"].item()) != state["step"]
        ):
            raise ValueError("Memory scorer resume identity or Adam step differs")
    history = state["history"]
    step = saved = state["step"]
    size = s["logical_batch"]
    try:
        for i in range(step, s["updates"]):
            begin = perf_counter()
            grad = None
            losses = []
            for row in items[i * size : (i + 1) * size]:
                b.check()
                embedded = mx.stop_gradient(
                    model.text.model.embed_tokens(mx.array(row["prompt_ids"]))
                )
                v, g = value_grad(scorer, embedded, row["gold_start"])
                mx.eval(v, g)
                losses.append(v.item())
                grad = g if grad is None else tree_map(lambda x, y: x + y, grad, g)
            grad = tree_map(lambda x: x / size, grad)
            grad, norm = optim.clip_grad_norm(grad, 1.0)
            if not mx.isfinite(norm).item():
                raise FloatingPointError("Nonfinite memory gradient")
            opt.update(scorer, grad)
            mx.eval(scorer.parameters(), opt.state)
            history.append(
                {"loss": sum(losses) / size, "seconds": perf_counter() - begin}
            )
            state["step"] = i + 1
            if state["step"] % 16 == 0:
                checkpoint.write(cp, scorer, opt, state)
                saved = state["step"]
    finally:
        if state["step"] != saved:
            checkpoint.write(cp, scorer, opt, state)
    gen = checkpoint_location(cp)
    save(
        out / "reader-recovery.json",
        {
            "generation": str(gen),
            "weights": digest(gen / "adapters.safetensors"),
            "optimizer": digest(gen / "optimizer.safetensors"),
            "updates": state["step"],
        },
    )
    scorer.save_weights(str(out / "attention-reader.safetensors"))
    mx.save_safetensors(
        str(out / "reader-optimizer.safetensors"), dict(tree_flatten(opt.state))
    )
    save(out / "training.json", history)
    return history


def score(m, t, rows, s, b, out):
    report = []
    for row in rows:
        v = ledger_curriculum.measure(m, t, row, s["generation_cap"], b)
        sel = v.get("attention_memory")
        if sel:
            v["located"] = (
                sel["start"] <= row["gold_start"] and sel["stop"] >= row["gold_stop"]
            )
        report.append(v)
        packed(out, {"records": report})
    return report


def run(freeze):
    s = read(REG)
    t = load_tokenizer(local_snapshot())
    train, cal, panel = memory_data.dataset(t, s)
    material = {
        "training_digest": fingerprint(train),
        "training_examples": len(train),
        "calibration": cal,
        "panel": panel,
    }
    if freeze and not DATA.exists():
        save(DATA, material)
    if read(DATA) != material:
        raise ValueError("Frozen memory data differs")
    if freeze:
        return
    if digest(DATA) != read("research/registry.json")["datasets"][str(DATA)]["sha256"]:
        raise ValueError("Memory panel digest differs")
    parent = Path(s["parent"])
    if digest(parent / "adapters.safetensors") != s["parent_sha256"]:
        raise ValueError("Reader backbone differs")
    root = core.OUT / s["id"]
    if (root / "decision.json").exists():
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True)
    core.assets()
    b = core.Budget(s["id"], s["limits"])
    result = {
        "passed": False,
        "registration": digest(REG),
        "data": digest(DATA),
        "parent": s["parent_sha256"],
    }
    try:
        m, t, _ = load_trained(parent)
        mx.random.seed(s["seed"])
        arch = {
            "width": s["width"],
            "query_length": s["query_length"],
            "span_width": s["span_width"],
            "stride": s["stride"],
        }
        scorer = PassageScorer(m.args.hidden_size, **arch)
        history = fit(s, m, scorer, train, b, root)
        scorer.eval()
        result["update_seconds"] = sum(h["seconds"] for h in history)
        result["scorer_weights"] = digest(root / "attention-reader.safetensors")
        candidates = {}
        for gain in s["gains"]:
            object.__setattr__(
                m,
                "attention_reader",
                AttentionReader(scorer, gain, s["activation_length"]),
            )
            rows = score(m, t, cal, s, b, root / f"calibration-{gain}.json.gz")
            candidates[gain] = {
                "correct": sum(v["correct"] for v in rows),
                "located": sum(v["located"] for v in rows),
                "seconds": sum(v["elapsed_s"] for v in rows),
            }
        best = max(s["gains"], key=lambda g: (candidates[g]["correct"], -g))
        result.update(calibration=candidates, gain=best)
        if candidates[best]["correct"] < 8 or candidates[best]["located"] < 10:
            result.update(status="calibration_failed")
        else:
            model = root / "model"
            model.mkdir(exist_ok=True)
            for name in ("state.json", "adapters.safetensors"):
                shutil.copyfile(parent / name, model / name)
            shutil.copyfile(
                root / "attention-reader.safetensors",
                model / "attention-reader.safetensors",
            )
            meta = {
                "architecture": arch,
                "gain": best,
                "minimum": s["activation_length"],
                "scorer_sha256": result["scorer_weights"],
                "adapter_sha256": s["parent_sha256"],
            }
            save(model / "attention-reader.json", meta)
            result["weights"] = fingerprint(meta)
            del m, t, scorer
            gc.collect()
            mx.clear_cache()
            reports = {}
            for arm in ("base", "candidate"):
                m, t, _ = base() if arm == "base" else load_trained(model)
                reports[arm] = score(m, t, panel, s, b, root / (arm + ".json.gz"))
                if arm == "candidate":
                    original = unpack(parent.parent / "candidate.json.gz")["records"]
                    _, math_items, _ = rationale.material(read(rationale.REG))
                    # Exercise the production generator including its short-prompt route.
                    short = []
                    for item, old in zip(math_items, original, strict=True):
                        v = ledger_curriculum.measure(m, t, dict(item, seed=0), 256, b)
                        short.append(v)
                        if (
                            v["token_ids"] != old["token_ids"]
                            or "attention_memory" in v
                        ):
                            raise ValueError("Short-capability parity regression")
                    packed(root / "short.json.gz", {"records": short})
                    result["short_parity"] = len(short)
                    result["retention"] = bootstrap.retention(m, 2, b)
                del m, t
                gc.collect()
                mx.clear_cache()
            r = reports["candidate"]
            a, c = (ledger_curriculum.scores(reports[k]) for k in ("base", "candidate"))
            result.update(
                status="completed",
                scores={"base": a, "candidate": c},
                located=sum(v["located"] for v in r),
            )
            result["passed"] = (
                c["correct"] >= 7
                and sum(v["correct"] for v in r[4:]) >= 3
                and all(
                    any(
                        r[i]["correct"] and panel[i]["style"] == style
                        for i in range(4, 8)
                    )
                    for style in ("table", "filing")
                )
                and c["correct"] - a["correct"] >= 2
                and c["seconds"] <= 1.75 * a["seconds"]
                and result["retention"]["nll"]
                <= read(parent.parent / "decision.json")["base"]["retention"]["nll"]
                + 0.03
            )
    except (MemoryError, TimeoutError, InterruptedError) as exc:
        result.update(status="interrupted", reason=str(exc))
    finally:
        b.close()
    save(root / "decision.json", result)
    print(result["status"], result["passed"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", action="store_true")
    run(p.parse_args().freeze)
