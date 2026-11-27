"""Protect the source-qualification and real-view boundaries exercised by the native fit."""

import json
from collections import Counter
from pathlib import Path

from loopqwen.io import read
from loopqwen.research.proof_curriculum import qualification, turns
from scripts.proof_curriculum_reading import references

SPEC = Path("research/experiments/proof-curriculum-01.json")


def test_qualification_is_the_source_inventory_and_turns_are_examples():
    spec = read(SPEC)
    census = qualification(spec)
    assert census["distinct_worlds"] == 192
    assert census["verdicts"] == {"true": 64, "false": 64, "uncertain": 64}
    assert census["multihop_known"] == 118
    assert (census["human"], census["packages"], census["version_checks"]) == (
        384,
        10,
        70,
    )
    examples = turns(spec, None)
    types = Counter(r["kind"] for r in examples)
    assert types == {"inline": 646, "file": 646, "opening": 646, "rehearsal": 128}
    material = read(spec["training"])
    for key in ("human", "package", "proof"):
        row = material[key][0]
        case = {r["kind"]: r for r in examples if r.get("case") == row["id"]}
        assert case["inline"]["target"] == case["file"]["target"]
        assert case["file"]["target"]["reasoning_content"] == row["reason"]
        assert case["file"]["messages"][-1]["role"] == "tool"

        assert json.loads(case["file"]["messages"][-1]["content"]) == {
            "value": row["evidence"]
        }
        assert case["opening"]["target"] == case["file"]["messages"][-2]


def test_numbered_citations_exclude_versions_and_preserve_invalid_indices():
    assert references(
        "Premises 2, 3 and 5 support [6, 8]. Python 3.10.5 occurs in 2026."
    ) == [2, 3, 5, 6, 8]
    assert references(
        "Rule 0 needs [21]. A textual paraphrase alone has no number."
    ) == [0, 21]
    assert not references("The file lists version 3.8 as the minimum.")
