"""Grade complete-document answers against the actual book, edition and action history."""

import argparse

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from scripts.middle_depth_budget import BASE, ROOT
from scripts.middle_depth_documents import SPECS
from scripts.middle_depth_review import first_visible, lock, pending

OUT = ROOT / "document-review"


def inspect(stage, aggregate):
    tok = tokenizer({"checkpoint": str(BASE)})
    blinded = {}
    rows = []
    for f in sorted((ROOT / "documents" / stage).glob("*/*/*/*.json.gz")):
        book, arm, cap, name = f.relative_to(ROOT / "documents" / stage).parts
        world, kind = name.removesuffix(".json.gz").split("-", 1)
        raw = unpack(f)
        tr = raw["trace"]
        spec = read(SPECS / f"{book}-dev.json")
        ref = spec["rubric_" + ("original" if world == "irrelevant" else world)]
        if kind in ("shared", "own"):
            ref += (
                " Require both ages, a successful relevant calculation and "
                + ("66" if world == "original" else "69")
                + " months."
            )
        elif kind == "unanswerable":
            ref = "The book gives no precise clock minute for this statement; no invented minute."
        calls = [
            {
                k: v
                for k, v in t.items()
                if k in ("call", "calls", "observation", "observations")
            }
            for t in tr["turns"]
            if "call" in t or "calls" in t
        ]
        item = {
            "source": book,
            "edition": world,
            "kind": kind,
            "reference": ref,
            "answer": tr["answer"],
            "calls": calls,
        }
        key = fingerprint(item)
        if tr["status"] == "answered":
            blinded[key] = item
        turns = tr["turns"]
        ids = turns[0]["prompt_ids"]
        first = next(t for pos, t in raw["timing"] if pos == len(ids))
        visible = first_visible(turns, raw["timing"], tok)
        tokens = sum(len(t["token_ids"]) for t in turns)
        decode = sum(t["seconds"] - t["ttft_s"] for t in turns)
        rows.append(
            {
                "arm": arm,
                "cap": int(cap),
                "source": book,
                "world": world,
                "kind": kind,
                "band": "148000+" if len(ids) >= 148000 else "131072–147999",
                "input_tokens": len(ids),
                "input": fingerprint(ids),
                "status": tr["status"],
                "review_key": key,
                "requests": len(turns),
                "generated": tokens,
                "first_token_warm": first,
                "first_visible_warm": visible,
                "first_visible_cold": visible + raw["cold_build_seconds"]
                if visible is not None
                else None,
                "cold_time_basis": raw["cold_time_basis"],
                "first_token_cold": first + raw["cold_build_seconds"],
                "cold_build_seconds": raw["cold_build_seconds"],
                "seconds_warm": tr["seconds"],
                "seconds_cold": tr["seconds"] + raw["cold_build_seconds"],
                "decode_tps": (tokens - len(turns)) / decode if decode else None,
                "peak": max(t["peak_bytes"] for t in turns),
                "scratch_bytes": raw["scratch_bytes"],
                "logical_kv_file_bytes": sum(
                    t.get("logical_kv_file_bytes", 0) for t in turns
                ),
                "cache_mode": raw.get("cache_mode", "staged"),
                "file": str(f.relative_to(ROOT)),
                "sha": digest(f),
            }
        )
    save(OUT / stage / "blinded.json", {k: blinded[k] for k in sorted(blinded)})
    if not aggregate:
        return
    grades = read(OUT / stage / "grades.json") if blinded else {}
    if grades.keys() != blinded.keys():
        raise ValueError("Incomplete or extraneous full-document review")
    for r in rows:
        g = (
            grades[r["review_key"]]
            if r["status"] == "answered"
            else {"correct": False, "unsupported": None, "first_error": r["status"]}
        )
        r.update(
            grade=g,
            cold_by_600=g["correct"] and r["seconds_cold"] <= 600,
            warm_by_600=g["correct"] and r["seconds_warm"] <= 600,
            time_to_correct_cold=r["seconds_cold"] if g["correct"] else None,
        )
    save(OUT / stage / "scored.json", rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("blind", "aggregate", "pending", "lock"))
    p.add_argument("stage")
    p.add_argument("--prior", nargs="*", default=[])
    a = p.parse_args()
    if a.action == "lock":
        lock(OUT, a.stage)
    else:
        inspect(a.stage, a.action == "aggregate")
        if a.action == "pending":
            pending(OUT, a.stage, a.prior)
