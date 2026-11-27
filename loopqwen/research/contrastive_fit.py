"""Fit a residual-increment decoder on full responses and bidirectional evidence changes."""

import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten, tree_map

from scripts.contrastive_world_material import REASON, TABLES

from .. import checkpoint, qwen_tools
from ..conditional_reentry import resident
from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import checkpoint_location, digest, read, save
from .conditional_features import CACHE, feature_root
from .conditional_fit import batches, load_batch, probes
from .contrastive_features import FIT, OLD, SOURCES
from .contrastive_grading import attested, selected_choice
from .core import Budget
from .cross_view_study import tokenizer
from .file_evidence_environment import capture
from .native_function_data import token_pair
from .registered import registered
from .table_policy_learning import visible
from .table_reentry_fit import accumulate, apply, root_for
from .table_worlds import FILENAME, prepare_tables, query_results

RESULTS = Path("research/results")


def index(s):
    rows = []
    archives = {}
    for name in s["feature_stages"]:
        root = RESULTS / name
        if not read(root / "decision.json")["passed"]:
            raise ValueError("A required native feature stage did not pass: " + name)
        assert read(root / "backbone.json") == read(OLD / "backbone.json")
        archives[name] = digest(root / "checksums.json")
        where = feature_root(registered(name))
        for r in unpack(root / "feature-index.json.gz"):
            rows.append({**r, "file": str(where / r["file"])})
    assert len(rows) == len({r["id"] for r in rows})
    return rows, archives


def objective(net, s):
    def loss(readout, f, labels, positive, pairs):
        ordinary = f[:, 0 if s["passes"] == 2 else -1, :]
        teacher = mx.stop_gradient(
            nn.log_softmax(
                net.project(net.text.model.norm(ordinary)).astype(mx.float32)
            )
        )
        learned = nn.log_softmax(
            net.project(net.text.model.norm(readout(f))).astype(mx.float32)
        )
        nll = -mx.take_along_axis(learned, labels[:, None], axis=-1)[:, 0]
        ce = (nll * positive).sum()
        kl = (mx.sum(mx.exp(teacher) * (teacher - learned), axis=-1) * positive).sum()
        pair = (
            nn.softplus(s["margin"] - pairs @ nll).sum()
            if pairs.shape[0]
            else mx.array(0.0)
        )
        return ce + s["teacher_kl"] * kl + s["preference_weight"] * pair, ce, kl, pair

    return loss


def optimize(net, root, s, idx, guard):
    where = CACHE / s["id"]
    where.mkdir(parents=True, exist_ok=True)
    steps = batches(idx, s)
    opt = optim.AdamW(s["lr"], weight_decay=0.01)
    grad = nn.value_and_grad(net.readout, objective(net, s))
    registration = digest(root / "registration.json")
    state = {"registration": registration, "step": 0, "log": []}
    if (where / "latest.json").exists():
        state = checkpoint.restore(where, net.readout, opt)
        assert state["registration"] == registration
    for step, micros in enumerate(steps, 1):
        if step <= state["step"]:
            continue
        g = None
        measure = [0.0] * 4
        count = targets = prefs = 0
        for micro in micros:
            guard.check()
            inputs, n = load_batch(idx, Path("."), micro)
            values, derivative = grad(net.readout, *inputs)
            g = accumulate(g, derivative, 1)
            mx.eval(g, values)
            measure = [a + v.item() for a, v in zip(measure, values, strict=True)]
            count += len(micro)
            targets += n
            prefs += inputs[-1].shape[0]
        if not all(map(math.isfinite, measure)):
            raise InterruptedError("Nonfinite native objective")
        norm = apply(net.readout, opt, tree_map(lambda v, n=targets: v / n, g))
        if norm <= 0:
            raise InterruptedError("Zero native gradient at a scheduled update")
        state["step"] = step
        state["log"].append(
            {
                "step": step,
                "turns": count,
                "targets": targets,
                "preferences": prefs,
                "objective_sum": measure[0],
                "nll_sum": measure[1],
                "teacher_kl_sum": measure[2],
                "preference_loss_sum": measure[3],
                "gradient_norm": norm,
            }
        )
        if step % 64 == 0 or step == len(steps):
            checkpoint.write(where, net.readout, opt, state)
            print("Increment update", step, "of", len(steps), flush=True)
    saved = checkpoint_location(where)
    shutil.copyfile(saved / "adapters.safetensors", root / "readout.safetensors")
    shutil.copyfile(saved / "optimizer.safetensors", root / "optimizer.safetensors")
    return state["log"]


def greedy(net, tok, row, cap, guard):
    p, _ = token_pair(tok, row)
    ids = mx.array([p])
    tokens = []
    cache = net.make_cache()
    stops = set(tok.eos_token_ids) | {tok.convert_tokens_to_ids("<|im_end|>")}
    for j in range(cap):
        if j % 16 == 0:
            guard.check()
        token = mx.argmax(net.project(net.last_hidden(ids, cache))[0, -1]).item()
        tokens.append(token)
        if token in stops:
            break
        ids = mx.array([[token]])
    content, closed = visible(tok, {"tokens": tokens}, stops)
    return {
        "id": row["id"],
        "group": row["group"],
        "pair": row["pair"],
        "opening": p,
        "tokens": tokens,
        "visible": content,
        "closed": closed,
    }


def selection(rows, s):
    chosen = []
    for group, n in (("operator", 24), ("qualitative", 12), ("service_reply", 12)):
        pairs = defaultdict(list)
        for r in rows:
            if r["group"] == group:
                pairs[r["pair"]].append(r)
        keys = sorted(
            pairs,
            key=lambda key: fingerprint(
                [s["fit_seed"], "free action or reply", group, key]
            ),
        )[:n]
        chosen.extend(r for k in keys for r in pairs[k])
    return chosen


def grade(free, row, worlds, qualitative, folder):
    text = free["visible"]
    ok = free["closed"]
    group = row["group"]
    if group == "operator":
        w = worlds[row["pair"]]
        definitions = {v["function"]["name"]: v["function"] for v in row["tools"]}
        try:
            calls = qwen_tools.parse(text, definitions) if ok else []
        except (ValueError, TypeError):
            calls = []
        arg = (
            calls[0]["arguments"]
            if len(calls) == 1 and calls[0]["name"] == "database_query"
            else {}
        )
        statement = (
            arg.get("statement")
            if set(arg) == {"database_name", "statement"}
            and arg["database_name"] == FILENAME
            else None
        )
        queried = (
            query_results(statement, w, folder) if isinstance(statement, str) else []
        )
        return {
            "correct": bool(queried) and all(r["correct"] for r in queried),
            "statement": statement,
            "worlds": queried,
        }
    if group == "qualitative":
        result = (
            selected_choice(text, qualitative[row["pair"]]["members"][0]["choices"])
            if ok
            else None
        )
        return {
            "correct": result == row["target"]["content"],
            "choice": result,
            "expected": row["target"]["content"],
        }
    assert group == "service_reply"
    results = json.loads(row["messages"][-1]["content"])
    return {
        "correct": bool(ok and attested(text, [r.values() for r in results])),
        "original_reply": row["target"]["content"],
    }


def free_probes(root, net, tok, s, guard):
    rows = selection(unpack(Path(s["training"])), s)
    worlds = {r["id"]: r for r in unpack(TABLES / "training.json.gz")}
    qualitative = {r["id"]: r for r in unpack(REASON / "training.json.gz")}
    checks = []
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        prepare_tables(
            folder, [worlds[w["pair"]] for w in rows if w["group"] == "operator"]
        )
        for row in rows:
            rec = greedy(
                net,
                tok,
                row,
                s["free_sql_cap"]
                if row["group"] == "operator"
                else s["free_answer_cap"],
                guard,
            )
            rec["score"] = grade(rec, row, worlds, qualitative, folder)
            checks.append(rec)
    packed(root / "training-free-generation.json.gz", checks)
    return checks


def summarize(s, log, likelihood, free):
    groups = defaultdict(list)
    for r in likelihood:
        groups[r["group"]].append(r["ordinary"] - r["trained"])
    means = {k: mean(v) for k, v in groups.items()}
    whole = {}
    counts = {}
    for g, n, width in (
        ("operator", 24, 1),
        ("qualitative", 12, 2),
        ("service_reply", 12, 2),
    ):
        pairs = defaultdict(list)
        for r in free:
            if r["group"] == g:
                pairs[r["pair"]].append(r["score"]["correct"])
        assert len(pairs) == n and all(len(v) == width for v in pairs.values())
        whole[g] = sum(all(v) for v in pairs.values())
        counts[g] = sum(sum(v) for v in pairs.values())
    distinct = len(
        {
            r["score"]["statement"]
            for r in free
            if r["group"] == "operator" and r["score"]["correct"]
        }
    )
    labels = sum(r["targets"] for r in log)
    prefs = sum(r["preferences"] for r in log)
    gate = {
        "complete": labels == s["expected_positions"]
        and prefs == s["expected_preferences"],
        "gradients": all(
            math.isfinite(r["gradient_norm"]) and r["gradient_norm"] > 0 for r in log
        ),
        "collapse": len(likelihood) == 112
        and all(r["collapsed_equal"] for r in likelihood),
        "balanced_likelihood": len(means) == 14 and mean(means.values()) >= 0.08,
        "broad_likelihood": min(
            means[k] for k in ("instruction", "arithmetic", "explanation", "legacy")
        )
        >= -0.02,
        "unassisted_sql": whole["operator"] >= 8 and distinct >= 6,
        "paired_qualitative": whole["qualitative"] >= 3,
        "paired_service_reply": whole["service_reply"] >= 3,
    }
    return {
        "passed": all(gate.values()),
        "gates": gate,
        "supervised_positions": labels,
        "preferences": prefs,
        "updates": len(log),
        "mean_nll_gain": means,
        "free_whole": whole,
        "free_correct": counts,
        "free_distinct_sql": distinct,
    }


def run():
    s = registered(FIT)
    idx, parents = index(s)
    root = root_for(s)
    capture(
        root,
        (
            *SOURCES,
            "loopqwen/research/contrastive_fit.py",
            "loopqwen/research/contrastive_grading.py",
            "loopqwen/research/conditional_fit.py",
        ),
    )
    guard = Budget(FIT, s["limits"])
    tok = tokenizer(s)
    try:
        for r in idx:
            guard.check()
            assert digest(Path(r["file"])) == r["sha256"]
        ident = read(OLD / "backbone.json")
        net = resident(s["checkpoint"], root, s, ident)
        log = optimize(net, root, s, idx, guard)
        packed(root / "training-log.json.gz", log)
        fit = {
            "registration": digest(root / "registration.json"),
            "weights": digest(root / "readout.safetensors"),
            "backbone": ident,
            "feature_archives": parents,
            "trainable": sum(v.size for _, v in tree_flatten(net.readout.parameters())),
            "updates": len(log),
        }
        save(root / "fit.json", fit)
        probe = probes(net, idx, Path("."), s)
        by = {r["id"]: r for r in idx}
        for r in probe:
            state = mx.load(by[r["id"]]["file"])["positive"]
            ordinary = state[:, 0 if s["passes"] == 2 else -1]
            collapsed = mx.broadcast_to(ordinary[:, None, :], state.shape)
            r["collapsed_equal"] = mx.array_equal(
                net.readout(collapsed), ordinary
            ).item()
        save(root / "training-probes.json", probe)
        free = free_probes(root, net, tok, s, guard)
        d = summarize(s, log, probe, free)
    except (TimeoutError, MemoryError, InterruptedError) as e:
        d = {"passed": False, "status": "interrupted", "reason": str(e)}
        where = CACHE / FIT
        if (where / "latest.json").exists():
            step = checkpoint_location(where)
            for fn in ("adapters.safetensors", "optimizer.safetensors", "state.json"):
                shutil.copyfile(step / fn, root / ("partial-" + fn))
            d["durable_updates"] = read(step / "state.json")["step"]
    finally:
        guard.close()
    save(root / "decision.json", d)


if __name__ == "__main__":
    run()
