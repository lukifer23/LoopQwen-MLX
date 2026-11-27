"""Read both questions about one situation and both decisions at a fixed dependency version."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save

PANEL = Path("research/data/visit-residual-panel.json")
PLAN = Path("research/diagnostics/visit-residual-reading.json")
ROOT = Path("research/results/visit-residual-01")
DOC = Path("docs/measurements")
MASK = DOC / "visit-residual-blinded-answers.json"
RATINGS = DOC / "visit-residual-blinded-ratings.json"
VERDICTS = ("supported", "partial", "wrong", "absent")


def plan(rows):
    clusters = {r["cluster"] for r in rows if r["family"] == "effects"}
    sample = sorted(clusters, key=lambda c: fingerprint([239803, c]))[:12]
    selected = [
        r["id"]
        for r in rows
        if (r["family"] == "effects" and r["cluster"] in sample)
        or r["family"] == "same_version"
    ]
    assert len(clusters) == 24 and len(selected) == 32
    return {
        "panel": digest(PANEL),
        "selected": selected,
        "human_situations": sample,
        "verdicts": list(VERDICTS),
        "method": "Both original questions on twelve deterministically sampled situations and all eight fixed-version changes of wheel. Mask the two systems, token counts, timings and automatic scores. Judge the visible answer against the supplied human source or the independently verified wheel metadata and decision. A supported answer must resolve the requested conclusion. Credit meaningful progress as partial, score an opposite conclusion as wrong, and count absent final answers. Separately record unsupported claims. Seal the signed-off ratings before reading either system's held totals.",
    }


def build(mode):
    rows = read(PANEL)["rows"]
    planned = plan(rows)
    if mode == "plan":
        save(PLAN, planned)
        return
    assert read(PLAN) == planned
    masked, keys = {}, {}
    for arm in ("candidate", "control"):
        cases = unpack(ROOT / (arm + ".json.gz"))
        for w, c in zip(rows, cases, strict=True):
            assert c["id"] == w["id"]
            if w["id"] not in planned["selected"]:
                continue
            (r,) = c["requests"]
            if w["family"] == "effects":
                evidence = {"published_answers": w["answers"]}
            else:
                evidence = {
                    "source": w["source"],
                    "dependency": w["dependency"],
                    "requirement": w["requirement"],
                    "version": w["released_version"],
                    "allowed": w["allowed"],
                    "original_header": w["current"]["observation"],
                }
            ob = {
                "request": w["questions"][0],
                "reference": evidence,
                "visible_answer": r["trajectory"]["answer"] or "",
            }
            key = fingerprint(ob)[:20]
            assert key not in masked or masked[key] == ob
            masked[key] = ob
            keys[arm, w["id"]] = key
    masked = dict(sorted(masked.items()))
    if mode == "blind":
        if MASK.exists():
            assert read(MASK) == masked
        else:
            save(MASK, masked)
        return
    assert read(MASK) == masked
    ratings = read(RATINGS)
    assert ratings.keys() == masked.keys()
    counts = {}
    for arm in ("candidate", "control"):
        by_family = defaultdict(Counter)
        grouped = defaultdict(list)
        for w in rows:
            if w["id"] not in planned["selected"]:
                continue
            rec = ratings[keys[arm, w["id"]]]
            assert rec["verdict"] in VERDICTS and rec["reason"].strip()
            assert type(rec["unsupported_claim"]) is bool
            by_family[w["family"]][rec["verdict"]] += 1
            by_family[w["family"]]["unsupported_claim"] += rec["unsupported_claim"]
            grouped[w["family"], w["cluster"]].append(rec["verdict"])
        for (family, cluster), vv in grouped.items():
            assert len(vv) == 2
            by_family[family]["both_supported"] += all(v == "supported" for v in vv)
        counts[arm] = {k: dict(v) for k, v in by_family.items()}
    save(DOC / "visit-residual-reading.json", counts)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "blind", "report"))
    build(p.parse_args().action)
