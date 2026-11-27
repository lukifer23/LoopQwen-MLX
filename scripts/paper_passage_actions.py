"""Count paired question sensitivity and action supervision from sealed human-source records."""

import csv
import re
from collections import Counter, defaultdict
from fractions import Fraction

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read
from loopqwen.research.native_function_study import executions

# Match printed decimal quantities including grouped thousands. Do not infer number words or units.
PRINTED_NUMBER = re.compile(
    r"(?<![\d.,])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\d)"
)


def make(paper, root, spec):
    fit = read(spec["training"])
    groups = {p["id"]: p["kind"] for p in fit["pairs"]}
    counts = defaultdict(Counter)
    for row in fit["examples"]:
        if "pair" not in row:
            continue
        assert [m["role"] for m in row["messages"]] == ["user"]
        c = counts[groups[row["pair"]]]
        c["openings"] += 1
        calls = row["target"].get("tool_calls", [])
        c["with_calculator"] += any(t["function"]["name"] == "calculate" for t in calls)

    panel = read(spec["material"])["rows"]
    with (paper / "passage-action-requests.csv").open("w", newline="") as f:
        out = csv.writer(f)
        out.writerow(
            [
                "arm",
                "passage_id",
                "question_id",
                "reference",
                "reference_printed_in_passage",
                "executed_calculator",
                "completed",
                "correct",
                "unchanged_visible_answer_with_other_question",
            ]
        )
        for arm in ("candidate", "control"):
            records = unpack(root / (arm + ".json.gz"))
            by_passage = defaultdict(list)
            for row, record in zip(panel, records, strict=True):
                assert row["id"] == record["id"]
                if row["family"] not in ("ASDiv", "DROP"):
                    continue
                request = record["requests"][0]
                tr = request["trajectory"]
                called = any(
                    event["call"]["name"] == "calculate" for event in executions(tr)
                )
                c = counts[arm + " " + row["family"]]
                c["openings"] += 1
                c["with_calculator"] += called
                if row["family"] != "DROP":
                    continue
                passage, _ = row["questions"][0].rsplit("\n\n", 1)
                numbers = {
                    Fraction(m[0].replace(",", ""))
                    for m in PRINTED_NUMBER.finditer(passage)
                }
                visible = Fraction(row["gold"]) in numbers
                bucket = counts[
                    arm
                    + (
                        " DROP reference printed"
                        if visible
                        else " DROP reference unprinted"
                    )
                ]
                bucket["openings"] += 1
                bucket["correct"] += request["score"]["correct_answer"]
                bucket["with_calculator"] += called
                by_passage[row["cluster"]].append((row, request, called, visible))
            for pid, pair in by_passage.items():
                assert len(pair) == 2 and pair[0][0]["gold"] != pair[1][0]["gold"]
                answers = [entry[1]["trajectory"]["answer"] for entry in pair]
                same = (
                    all(answers)
                    and answers[0].strip().casefold() == answers[1].strip().casefold()
                )
                c = counts[arm + " DROP pairs"]
                c["openings"] += 1
                c["both_visible"] += all(answers)
                c["same_answer"] += same
                for row, request, called, visible in pair:
                    tr = request["trajectory"]
                    out.writerow(
                        [
                            arm,
                            pid,
                            row["id"],
                            row["gold"],
                            visible,
                            called,
                            tr["status"] == "answered",
                            request["score"]["correct_answer"],
                            same,
                        ]
                    )
    with (paper / "passage-action-summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "openings",
                "with_calculator",
                "correct",
                "both_visible",
                "same_answer",
            ],
        )
        writer.writeheader()
        for group, count in sorted(counts.items()):
            writer.writerow({"group": group, **count})
