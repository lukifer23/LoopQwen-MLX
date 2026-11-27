"""Expose exact verified arithmetic and its distinction from a correct real-world answer."""

import csv
import json
from fractions import Fraction

from loopqwen.calculator_tool import checked_calculate
from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read
from loopqwen.research.native_function_study import executions


def make(paper, results, result):
    observed = results / "callback-recovery-01"
    if (observed / "candidate.json.gz").exists():
        result(observed.name)
        panel = read(read(observed / "registration.json")["material"])["rows"]
        with (paper / "rejected-equations-replay.csv").open("w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "id",
                    "cohort",
                    "question",
                    "gold",
                    "original_call",
                    "checked_result",
                    "checked_gold_match",
                    "actual_recovered_values",
                    "actual_grounded_success",
                ]
            )
            for w, r in zip(panel, unpack(observed / "candidate.json.gz"), strict=True):
                tr = r["trajectory"]
                t = tr["turns"][0]
                if w["cohort"] == "retention" or "error" not in t.get(
                    "observation", {}
                ):
                    continue
                call = t["call"]["arguments"]
                try:
                    new = checked_calculate(**call)
                except (ValueError, SyntaxError, ZeroDivisionError):
                    new = None
                writer.writerow(
                    [
                        w["id"],
                        w["cohort"],
                        w["question"],
                        w["gold"],
                        json.dumps(call),
                        json.dumps(new),
                        int(
                            new is not None
                            and Fraction(new["value"]) == Fraction(w["gold"])
                        ),
                        json.dumps(
                            [
                                v["observation"]
                                for v in tr["turns"]
                                if v.get("recovery_active") and "observation" in v
                            ]
                        ),
                        int(bool(r["score"]["grounded"])),
                    ]
                )
    study = "checked-equations-field-01"
    root = results / study
    if not (root / "checksums.json").exists():
        return
    result(study)
    panel = read(read(root / "registration.json")["material"])["rows"]
    with (paper / "checked-equations-field.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "source",
                "arm",
                "question",
                "gold",
                "answer",
                "status",
                "full_calls_and_results",
                "answer_correct",
                "grounded",
                "first_seconds",
                "total_seconds",
                "peak_bytes",
            ]
        )
        for side in ("candidate", "control"):
            arm = unpack(root / (side + ".json.gz"))
            for w, r in zip(panel, arm, strict=True):
                q = r["requests"][0]
                tr = q["trajectory"]
                writer.writerow(
                    [
                        w["id"],
                        w["family"],
                        side,
                        w["questions"][0],
                        w["gold"],
                        tr["answer"],
                        tr["status"],
                        json.dumps(executions(tr)),
                        int(q["score"]["correct_answer"]),
                        int(q["score"]["success"]),
                        q["first_s"],
                        q["wall_s"],
                        q["peak"],
                    ]
                )
