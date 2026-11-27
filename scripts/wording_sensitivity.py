"""Freeze text-reuse strata before generation and score them only after sealed source reviews."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

DOC = Path("docs/measurements")
STUDIES = {
    "contrastive-worlds": (
        "contrastive-world-transfer-01",
        ("contrastive-services", "contrastive-qualitative"),
    ),
    "shared-query": ("shared-query-transfer-01", ("shared-query",)),
}


def build(study):
    panel = Path("research/sources") / study / "transfer-panel.json.gz"
    audit = DOC / (study + "-wording-overlap.json")
    flags = defaultdict(list)
    for r in read(audit)["fields"]:
        if r["whole"] and r["words"] >= 8 or r.get("matched_words", 0) >= 12:
            flags[r["id"]].append(r["field"])
    rows = unpack(panel)["rows"]
    marked = set()
    for w in rows:
        if flags.get(w.get("source")) or flags.get(w["id"]):
            marked.add((w["family"], w["cluster"]))
    output = []
    for w in rows:
        if w["family"] == "locale":
            group = "unassessed"
        elif (w["family"], w["cluster"]) in marked:
            group = "source contains repeated wording"
        else:
            group = "below wording threshold"
        output.append(
            {
                "id": w["id"],
                "family": w["family"],
                "cluster": w["cluster"],
                "stratum": group,
                "fields": flags.get(w["id"], flags.get(w.get("source"), [])),
            }
        )
    return {
        "stage": STUDIES[study][0],
        "panel": digest(panel),
        "audit": digest(audit),
        "rule": "A source is marked if any audited field is fully contained in a training field and has at least eight normalized words, or if twelve contiguous normalized words coincide. Keep both members of each registered source together. Full locale documents are outside this wording audit. All original product gates keep every registered item. Descriptive strata are fixed before held model generation or judgments. Public pretraining data cannot be tested here.",
        "counts": {
            f: dict(Counter(r["stratum"] for r in output if r["family"] == f))
            for f in sorted({r["family"] for r in output})
        },
        "rows": output,
    }


def run(study, mode, *, streamed=False, variant=None):
    variant = variant or ("streamed" if streamed else None)
    if variant and variant not in ("streamed", "elastic"):
        raise ValueError("Unknown registered comparison")
    if variant and (study != "contrastive-worlds" or mode != "report"):
        raise ValueError(
            "The streamed comparison inherits the frozen contrastive wording strata"
        )
    plan = Path("research/diagnostics") / (study + "-wording-strata.json")
    expected = build(study)
    if mode == "freeze":
        persist(plan, expected)
        print(study, expected["counts"])
        return
    committed(plan)
    assert expected == read(plan)
    stage, reviews = STUDIES[study]
    if variant:
        stage = variant + "-world-transfer-01"
        reviews = (variant + "-services", variant + "-qualitative")
    root = Path("research/results") / stage
    committed(root / "checksums.json")
    for prefix in reviews:
        committed(DOC / (prefix + "-reading.json"))
    streams = [unpack(root / (arm + ".json.gz")) for arm in ("candidate", "control")]
    cells = defaultdict(list)
    for r, a, b in zip(expected["rows"], *streams, strict=True):
        assert r["id"] == a["id"] == b["id"]
        for item in a, b:
            assert len(item["requests"]) == 1
        cells[r["family"], r["stratum"]].append(
            (
                r["cluster"],
                *[bool(q["requests"][0]["score"]["success"]) for q in (a, b)],
            )
        )
    report = []
    for (family, stratum), items in sorted(cells.items()):
        sources = defaultdict(list)
        for key, a, b in items:
            sources[key].append((a, b))
        report.append(
            {
                "family": family,
                "stratum": stratum,
                "items": len(items),
                "sources": len(sources),
                "candidate_correct": sum(a for _, a, _ in items),
                "base_correct": sum(b for _, _, b in items),
                "paired_gains": sum(a and not b for _, a, b in items),
                "paired_losses": sum(b and not a for _, a, b in items),
                "all_source_members_correct": [
                    sum(all(x[i] for x in members) for members in sources.values())
                    for i in (0, 1)
                ],
            }
        )
    save(
        DOC
        / ((variant + "-world" if variant else study) + "-wording-sensitivity.json"),
        {
            "registration": digest(plan),
            "primary": digest(root / "checksums.json"),
            "order": fingerprint([r["id"] for r in expected["rows"]]),
            "cells": report,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("study", choices=STUDIES)
    p.add_argument("mode", choices=("freeze", "report"))
    p.add_argument("--streamed", action="store_true")
    p.add_argument("--variant", choices=("streamed", "elastic"))
    a = p.parse_args()
    run(a.study, a.mode, streamed=a.streamed, variant=a.variant)
