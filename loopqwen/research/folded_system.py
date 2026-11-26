"""Measure a single instruction backbone with one genuinely repeated gated-delta layer."""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.utils import load_tokenizer

from ..evaluation import executable_tools, program_questions
from ..evaluation.mechanism import packed, unpack
from ..evaluation.mixed_field import catalog, grade
from ..evaluation.relations import cluster_randomization
from ..evaluation.statistics import exact_mcnemar
from ..folded_decoder import from_release
from ..identity import fingerprint
from ..io import digest, read, save
from ..model_loader import local_snapshot
from ..tool_compare import DenseAssistant
from ..tool_session import initial_messages
from .core import Budget
from .folded_fit import fit
from .mixed_field import replay_exchange
from .registered import registered


def tools_for(w, field, replay=False):
    if w["family"] in ("math", "judgment"):
        return []
    if w["family"] == "public":
        return executable_tools.tools_for(w)
    return catalog(w, field, replay)


def judge(w, tr, tok):
    group = w["family"]
    if group in ("package", "database", "file", "reading", "arithmetic"):
        return grade(w, tr, tok)
    done = tr["status"] == "answered"
    if group == "public":
        scored = executable_tools.score(w, tr)
        return {
            "success": scored["success"],
            "answer": scored["correct_answer"],
            "grounded": scored["success"],
            "completed": done,
        }
    if group == "judgment":
        from scripts.contrast_study import label

        correct = label(tok, tr["turns"][0]["token_ids"]) == w["gold"]
    else:
        correct = program_questions.score(w, tr)["correct_answer"]
    return {
        "success": correct and done,
        "answer": correct,
        "grounded": None,
        "completed": done,
    }


def construct(spec, root):
    actor, repair, removed = from_release(Path(spec["checkpoint"]), spec)
    meta = read(root / "fit.json")
    weights = root / "folded-repair.safetensors"
    if digest(weights) != meta["weights"] or meta["training"] != digest(
        Path(spec["training"])
    ):
        raise ValueError("Folded adapter identity differs")
    repair.load_weights(str(weights), strict=True)
    mx.eval(repair.parameters())
    del removed
    return actor


def native_checks(actor, w):
    plain, tools = actor.language, actor._tool_model
    i = plain.span.start
    assert (
        len(plain.text.layers) == len(tools.text.layers) == 23
        and len(plain.sites) == 24
        and len(tools.sites) == 25
    )
    assert all(
        id(v) == id(other)
        for a, b in zip(plain.text.layers[:-1], tools.text.layers[:-1], strict=True)
        for (_, v), (_, other) in zip(
            tree_flatten(a.parameters()), tree_flatten(b.parameters()), strict=True
        )
    )
    arrays = [
        {id(v): v.nbytes for _, v in tree_flatten(model.parameters())}
        for model in (plain, tools)
    ]
    combined = sum((arrays[0] | arrays[1]).values())
    shared = sum(arrays[0][k] for k in arrays[0].keys() & arrays[1].keys())
    vals = {}
    ids = w["initial_ids"][-24:]
    for name, m in (("plain", plain), ("tool", tools)):
        cache = m.make_cache()
        assert cache.slots[i, 0] is not cache.slots[i, 1]
        full = m.project(m.hidden(mx.array([ids]))[:, -1:])
        m.hidden(mx.array([ids[:-1]]), cache)
        split = m.project(m.hidden(mx.array([[ids[-1]]]), cache))
        vals[name] = {
            "unique_layers": len(m.text.layers),
            "model_parameter_bytes": sum(arrays[int(name == "tool")].values()),
            "combined_parameter_bytes": combined,
            "shared_parameter_bytes": shared,
            "calls": len(m.sites),
            "states": len(cache.slots),
            "first": int(full.argmax().item()),
            "incremental": int(split.argmax().item()),
            "max_logit_error": float(
                mx.abs(full.astype(mx.float32) - split.astype(mx.float32)).max().item()
            ),
            "shared_input_tokens": ids,
        }
    return vals


def family_scores(data, panel, prior):
    metrics = {}
    family = lambda w: w["family"]
    for group in ["all", "fresh", *sorted({family(w) for w in panel})]:
        ix = [
            i
            for i, w in enumerate(panel)
            if group in ("all", family(w))
            or group == "fresh"
            and family(w) in ("math", "judgment")
        ]
        cell = {}
        for side, rows in data.items():
            chosen = [rows[i] for i in ix]
            cell[side] = {
                "n": len(ix),
                **{
                    k: sum(bool(r["score"][k]) for r in chosen)
                    for k in ("success", "answer", "completed", "grounded")
                },
                "seconds": sum(r["wall_s"] for r in chosen),
                "first": sum(r["first_token_s"] for r in chosen),
                "peak": max(r["peak"] for r in chosen),
                "tokens": sum(
                    len(t["token_ids"])
                    for r in chosen
                    for t in r["trajectory"]["turns"]
                ),
            }
        diffs = [
            int(data["candidate"][i]["score"]["success"])
            - int(data["control"][i]["score"]["success"])
            for i in ix
        ]
        clusters = {}
        for i, v in zip(ix, diffs, strict=True):
            w = panel[i]
            k = w.get("path", w.get("table_id", w["id"]))
            clusters[k] = clusters.get(k, 0) + v
        cell.update(
            gains=diffs.count(1),
            losses=diffs.count(-1),
            p=exact_mcnemar(diffs.count(1), diffs.count(-1)),
            cluster_p=cluster_randomization(list(clusters.values())),
        )
        metrics[group] = cell
    retained = {group: [] for group in ("field", "public")}
    for w, r in zip(panel, data["candidate"], strict=True):
        if w["id"] in prior and prior[w["id"]] and not r["score"]["success"]:
            retained["public" if w["family"] == "public" else "field"].append(w["id"])
    return metrics, retained


def outcome(data, panel, hidden, parity, prior, spec):
    metrics, retained = family_scores(data, panel, prior)
    validation = {}
    for source in ("raw", "dialogue"):
        subset = [r for r in hidden if r["domain"] == source]
        validation[source] = {
            k: sum(r[k] for r in subset) / len(subset)
            for k in ("uncorrected", "skip", "learned")
        }
    a, b = metrics["all"]["candidate"], metrics["all"]["control"]
    passed = (
        a["success"] >= b["success"] + spec["net"]
        and metrics["all"]["p"] <= 0.05
        and metrics["all"]["cluster_p"] <= 0.05
        and metrics["fresh"]["gains"] - metrics["fresh"]["losses"] >= spec["fresh_gain"]
        and all(
            v["candidate"]["success"] >= v["control"]["success"]
            and v["candidate"]["answer"] >= v["control"]["answer"]
            and v["candidate"]["completed"] >= v["control"]["completed"]
            for v in metrics.values()
        )
        and not any(retained.values())
        and all(a[k] < b[k] for k in ("seconds", "first", "peak"))
        and all(
            v["learned"] <= 0.8 * v["uncorrected"] and v["learned"] < v["skip"]
            for v in validation.values()
        )
        and all(
            v["first"] == v["incremental"] and v["max_logit_error"] <= 0.6
            for v in parity.values()
        )
    )
    return {
        "passed": bool(passed),
        "groups": metrics,
        "historically_correct_now_lost": retained,
        "hidden_validation": validation,
        "cache_parity": parity,
    }


def old_scores(spec):
    scores = {}
    for k in ("public", "field"):
        p = Path(spec[k])
        assert digest(p / "checksums.json") == spec[k + "_sha256"]
        scores.update(
            {r["id"]: r["score"]["success"] for r in unpack(p / "candidate.json.gz")}
        )
    return scores


def verify(root, d):
    spec = read(root / "registration.json")
    meta = read(root / "fit.json")
    assert (
        meta["weights"] == digest(root / "folded-repair.safetensors")
        and meta["updates"] == spec["updates"]
        and meta["training"] == digest(Path(spec["training"]))
    )
    training = read(spec["training"])["rows"]
    hidden = unpack(root / "hidden-validation.json.gz")
    assert [r["id"] for r in hidden] == [
        w["id"] for w in training if w["split"] == "validation"
    ]
    assert meta["observed_training_positions"] == sum(
        len(w["positions"]) for w in training if w["split"] == "train"
    )
    tok = load_tokenizer(local_snapshot())
    panel = read(spec["material"])["rows"]
    field = read(spec["field_data"])
    data = {}
    for s in ("candidate", "control"):
        arm = unpack(root / (s + ".json.gz"))
        for w, r in zip(panel, arm, strict=True):
            tr = r["trajectory"]
            catalog_ = tools_for(w, field, True)
            user = w.get("user_request", w["question"])
            msgs = (
                initial_messages(user, catalog_)
                if catalog_
                else [{"role": "user", "content": user}]
            )
            assert (
                w["id"] == r["id"]
                and tr["turns"][0]["prompt_ids"] == w["initial_ids"]
                and r["input_sha256"] == fingerprint(w["initial_ids"])
            )
            replay_exchange(tok, catalog_, msgs, tr, spec)
            assert (
                r["score"] == judge(w, tr, tok)
                and r["wall_s"] >= tr["seconds"]
                and 0 < r["first_token_s"] <= r["wall_s"]
                and r["peak"] > 0
            )
        data[s] = arm
    parity = read(root / "parity.json")
    for mode, v in parity.items():
        assert (
            v["shared_input_tokens"] == panel[0]["initial_ids"][-24:]
            and v["unique_layers"] == 23
            and v["states"] == v["calls"] == (24 if mode == "plain" else 25)
        )
    a, b = parity["plain"], parity["tool"]
    assert a["combined_parameter_bytes"] == b["combined_parameter_bytes"]
    assert a["shared_parameter_bytes"] == b["shared_parameter_bytes"] > 0
    assert (
        a["model_parameter_bytes"]
        + b["model_parameter_bytes"]
        - a["shared_parameter_bytes"]
        == a["combined_parameter_bytes"]
    )
    assert d == outcome(data, panel, hidden, parity, old_scores(spec), spec)


def run(name):
    spec = registered(name)
    root = Path("outputs/research") / name
    root.mkdir(exist_ok=True, parents=True)
    if (root / "decision.json").exists():
        raise FileExistsError(root)
    release = Path(spec["release"])
    assert (
        digest(release / "checksums.json") == spec["release_sha256"]
        and read(release / "decision.json")["passed"]
    )
    assert digest(Path(spec["checkpoint"]) / "tool-export-files.json") == digest(
        release / "export-files.json"
    )
    assert all(
        digest(Path(spec["checkpoint"]) / name) == value
        for name, value in read(release / "export-files.json").items()
    )
    assert digest(Path(spec["field_data"])) == read(spec["material"])["field_sha256"]
    field = read(spec["field_data"])
    panel = read(spec["material"])["rows"]
    from ..package_tools import snapshot

    assert snapshot() == field["environment"] and all(
        digest(Path(p)) == v["sha256"] for p, v in field["files"].items()
    )
    prior = old_scores(spec)
    data = {}
    b = Budget(name, spec["limits"])
    try:
        if not (root / "fit.json").exists():
            actor, repair, removed = from_release(Path(spec["checkpoint"]), spec)
            fit(actor, repair, removed, spec, root, b)
            del actor, repair, removed
        for s in ("candidate", "control"):
            dest = root / (s + ".json.gz")
            if dest.exists():
                data[s] = unpack(dest)
                assert len(data[s]) == len(panel)
                continue
            gc.collect()
            mx.clear_cache()
            actor = None
            arm = []
            for w in panel:
                b.check()
                tools = tools_for(w, field)
                mx.synchronize()
                mx.reset_peak_memory()
                at = perf_counter()
                if actor is None:
                    actor = (
                        construct(spec, root)
                        if s == "candidate"
                        else DenseAssistant(False)
                    )
                length = len(w["initial_ids"])
                first = None

                def progress(consumed, n=length, started=at):
                    nonlocal first
                    if first is None and consumed == n:
                        first = perf_counter() - started
                    b.check()

                tr = actor.respond(
                    w.get("user_request", w["question"]),
                    tools,
                    max_tokens=spec["max_tokens"],
                    max_turns=spec["max_turns"],
                    reuse_prefix=False,
                    progress=progress,
                )
                mx.synchronize()
                elapsed = perf_counter() - at
                assert (
                    tr["turns"][0]["prompt_ids"] == w["initial_ids"]
                    and first is not None
                )
                arm.append(
                    {
                        "id": w["id"],
                        "input_sha256": fingerprint(w["initial_ids"]),
                        "trajectory": tr,
                        "score": judge(w, tr, actor.tokenizer),
                        "first_token_s": first,
                        "wall_s": elapsed,
                        "peak": mx.get_peak_memory(),
                    }
                )
                if len(arm) % 12 == 0:
                    print(
                        s, len(arm), sum(r["score"]["success"] for r in arm), flush=True
                    )
            if s == "candidate":
                save(root / "parity.json", native_checks(actor, panel[0]))
            packed(dest, arm)
            data[s] = arm
            del actor
        save(
            root / "decision.json",
            outcome(
                data,
                panel,
                unpack(root / "hidden-validation.json.gz"),
                read(root / "parity.json"),
                prior,
                spec,
            ),
        )
    finally:
        b.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    run(p.parse_args().name)
