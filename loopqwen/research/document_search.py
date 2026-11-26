"""Evaluate freely generated searches and answers on actual external text files."""

import argparse
import gc
from collections import Counter
from pathlib import Path
from time import perf_counter

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from ..evaluation.mechanism import packed, unpack
from ..evaluation.statistics import exact_mcnemar
from ..io import digest, read, save
from ..model_loader import local_snapshot
from ..tool_model import Assistant
from ..tool_session import initial_messages
from .bridge_reading import units
from .core import Budget
from .mixed_field import replay_exchange
from .registered import registered


def grade(w, trajectory):
    ans = units(trajectory["answer"] or "")
    refs = [units(s) for s in w["refs"]]
    done = trajectory["status"] == "answered"
    exact = done and any(ans == ref and ref for ref in refs)
    overlap = (
        max(
            (
                2
                * sum((Counter(ans) & Counter(ref)).values())
                / max(1, len(ans) + len(ref))
                for ref in refs
            ),
            default=0,
        )
        if done
        else 0
    )
    obs = [t["observation"] for t in trajectory["turns"] if "observation" in t]
    evidence = any(
        p["document"] == w["document"] and p["start"] <= a < b <= p["stop"]
        for v in obs
        for p in v.get("passages", [])
        for a, b in w["answer_ranges"]
    )
    return {
        "correct": bool(exact),
        "grounded": bool(exact and evidence),
        "evidence": evidence,
        "called": bool(obs),
        "completed": done,
        "f1": overlap,
    }


def outcome(panel, arms, index):
    groups = {}
    for family in ("all", "SQuAD", "Qasper"):
        ix = [i for i, w in enumerate(panel) if family in ("all", w["source"])]
        group = {}
        for side, rs in arms.items():
            rows = [rs[i] for i in ix]
            group[side] = {
                "n": len(ix),
                **{
                    k: sum(r["score"][k] for r in rows)
                    for k in ("correct", "grounded", "evidence", "called", "completed")
                },
                "f1": sum(r["score"]["f1"] for r in rows) / len(rows),
                "seconds": sum(r["seconds"] for r in rows),
                "first": sum(r["first"] for r in rows),
                "peak": max(r["peak"] for r in rows),
            }
        diff = [
            int(arms["candidate"][i]["score"]["grounded"])
            - int(arms["control"][i]["score"]["grounded"])
            for i in ix
        ]
        groups[family] = {
            **group,
            "gains": diff.count(1),
            "losses": diff.count(-1),
            "p": exact_mcnemar(diff.count(1), diff.count(-1)),
        }
    all_ = groups["all"]
    a, b = all_["candidate"], all_["control"]
    passed = (
        a["grounded"] >= max(12, b["grounded"] + 6)
        and all_["p"] <= 0.05
        and all(
            g["candidate"]["grounded"] >= g["control"]["grounded"] + 2
            and all(
                g["candidate"][k] >= g["control"][k]
                for k in ("correct", "f1", "completed")
            )
            for g in (groups["SQuAD"], groups["Qasper"])
        )
        and a["correct"] >= b["correct"] + 4
        and a["peak"] < b["peak"]
        and all(
            a[k] + index["candidate"] < b[k] + index["control"]
            for k in ("seconds", "first")
        )
    )
    return {"passed": bool(passed), "groups": groups, "index_seconds": index}


def verify(root, decision):
    from scripts.document_search_material import open_panel

    spec = read(root / "registration.json")
    mat = read(spec["material"])
    panel = mat["rows"]
    tok = load_tokenizer(local_snapshot())
    arms = {}
    with open_panel(mat) as tools:
        for side in ("candidate", "control"):
            rows = unpack(root / (side + ".json.gz"))
            for w, r in zip(panel, rows, strict=True):
                tr = r["trajectory"]
                assert (
                    w["id"] == r["id"]
                    and tr["turns"][0]["prompt_ids"] == w["initial_ids"]
                )
                replay_exchange(
                    tok, tools, initial_messages(w["request"], tools), tr, spec
                )
                assert (
                    grade(w, tr) == r["score"]
                    and 0 < r["first"] <= r["seconds"]
                    and r["peak"] > 0
                )
                for t in tr["turns"]:
                    expected = {(i, 0) for i in (3, 7, 11, 15, 19, 23)}
                    if side == "candidate":
                        expected.add((23, 1))
                    sites = t["kv_history"]
                    assert {(s["layer"], s["turn"]) for s in sites} == expected and all(
                        s["offset"]
                        == s["retained"]
                        == len(t["prompt_ids"]) + len(t["token_ids"]) - 1
                        for s in sites
                    )
            arms[side] = rows
    index = read(root / "index.json")
    assert all(index[s] > 0 for s in arms) and outcome(panel, arms, index) == decision


def run(name):
    from scripts.document_search_material import open_panel

    from ..tool_compare import DenseAssistant

    spec = registered(name)
    parent = Path(spec["requires"])
    bundle = Path(spec["checkpoint"])
    assert (
        digest(parent / "checksums.json") == spec["prerequisite_sha256"]
        and read(parent / "decision.json")["passed"]
    )
    assert all(
        digest(bundle / k) == h for k, h in read(parent / "export-files.json").items()
    )
    mat = read(spec["material"])
    root = Path("outputs/research") / name
    root.mkdir(parents=True, exist_ok=True)
    if (root / "decision.json").is_file():
        raise FileExistsError(root)
    budget = Budget(name, spec["limits"])
    arms = {}
    index_path = root / "index.json"
    index = read(index_path) if index_path.is_file() else {}
    try:
        for side in ("candidate", "control"):
            path = root / (side + ".json.gz")
            if path.exists():
                # Cold stream includes one model/index load per arm. Restart a partial
                # arm from its first request to keep its original timing definition.
                previous = unpack(path)
                if len(previous) == len(mat["rows"]):
                    arms[side] = previous
                    continue
            begin = perf_counter()
            with open_panel(mat) as tools:
                index[side] = perf_counter() - begin
                save(index_path, index)
                model = None
                rows = []
                for w in mat["rows"]:
                    budget.check()
                    mx.synchronize()
                    mx.reset_peak_memory()
                    start = perf_counter()
                    if model is None:
                        model = (
                            Assistant(bundle)
                            if side == "candidate"
                            else DenseAssistant(False)
                        )
                    observed = []

                    def progress(
                        consumed,
                        expected=len(w["initial_ids"]),
                        stamps=observed,
                        at=start,
                    ):
                        if consumed == expected and not stamps:
                            stamps.append(perf_counter() - at)
                        budget.check()

                    tr = model.respond(
                        w["request"],
                        tools,
                        max_tokens=spec["max_tokens"],
                        max_turns=spec["max_turns"],
                        reuse_prefix=False,
                        record_kv_history=True,
                        progress=progress,
                    )
                    mx.synchronize()
                    elapsed = perf_counter() - start
                    assert (
                        tr["turns"][0]["prompt_ids"] == w["initial_ids"]
                        and len(observed) == 1
                    )
                    rows.append(
                        {
                            "id": w["id"],
                            "trajectory": tr,
                            "score": grade(w, tr),
                            "first": observed[0],
                            "seconds": elapsed,
                            "peak": mx.get_peak_memory(),
                        }
                    )
                    if len(rows) % 8 == 0:
                        packed(path, rows)
                        print(
                            side,
                            len(rows),
                            sum(r["score"]["grounded"] for r in rows),
                            flush=True,
                        )
                arms[side] = rows
                del model
                gc.collect()
                mx.clear_cache()
        save(root / "decision.json", outcome(mat["rows"], arms, index))
    finally:
        budget.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    run(p.parse_args().name)
