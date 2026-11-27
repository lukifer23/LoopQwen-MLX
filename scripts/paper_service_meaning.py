"""Recount visible service answers with and without numbers inside resource names."""

import csv
import json

from loopqwen.evaluation.mechanism import unpack
from loopqwen.evaluation.service_answers import visible_reference
from loopqwen.io import read
from loopqwen.research.native_function_study import executions
from loopqwen.research.observation_review_study import paired


def operational(w, i, tr):
    action, gold = w["calls"][i], w["answers"][i]
    resource = next(iter(action["arguments"].values()))
    meaning = visible_reference(
        tr["answer"], resource, gold, file=w["family"] == "files"
    )
    events = executions(tr)
    correct = [
        e["call"] == action and str(next(iter(e["observation"].values()))) == gold
        for e in events
    ]
    return (
        tr["status"] == "answered" and meaning,
        len(correct) == 1 and all(correct),
        any(correct),
    )


def write(paper, results):
    studies = (
        "native-function-dialogue-01",
        "observation-review-01",
        "question-contrast-01",
    )
    with (
        (paper / "service-meaning-requests.csv").open("w", newline="") as raw,
        (paper / "service-meaning-summary.csv").open("w", newline="") as summary,
    ):
        rw, sw = csv.writer(raw), csv.writer(summary)
        rw.writerow(
            [
                "study",
                "arm",
                "family",
                "id",
                "exchange",
                "question",
                "reference",
                "answer",
                "calls",
                "registered_success",
                "meaning",
                "exact_call",
                "any_correct_call",
            ]
        )
        sw.writerow(
            [
                "study",
                "family",
                "metric",
                "loop",
                "base",
                "loop_gains",
                "loop_losses",
                "paired_p",
            ]
        )
        for name in studies:
            root = results / name
            panel = read(read(root / "registration.json")["material"])["rows"]
            successes = {}
            for side in ("candidate", "control"):
                groups = {}
                for w, r in zip(panel, unpack(root / (side + ".json.gz")), strict=True):
                    if w["family"] not in ("packages", "files"):
                        continue
                    grades = []
                    for i, q in enumerate(r["requests"]):
                        tr = q["trajectory"]
                        meaning, exact, any_call = operational(w, i, tr)
                        rw.writerow(
                            [
                                name,
                                side,
                                w["family"],
                                w["id"],
                                i + 1,
                                w["questions"][i],
                                w["answers"][i],
                                tr["answer"],
                                json.dumps(executions(tr), ensure_ascii=False),
                                int(q["score"]["success"]),
                                int(meaning),
                                int(exact),
                                int(any_call),
                            ]
                        )
                        grades.append(
                            (
                                q["score"]["success"],
                                meaning and exact,
                                meaning and any_call,
                                meaning,
                            )
                        )
                    for k, grade in zip(
                        (
                            "registered",
                            "meaning_exact_path",
                            "meaning_any_correct_call",
                            "meaning",
                        ),
                        zip(*grades),
                        strict=True,
                    ):
                        for family in (w["family"], "all"):
                            groups.setdefault((family, k), set())
                            if all(grade):
                                groups[family, k].add(w["id"])
                successes[side] = groups
            for (family, k), a in sorted(successes["candidate"].items()):
                b = successes["control"][family, k]
                pair = paired(a, b)
                sw.writerow(
                    [
                        name,
                        family,
                        k,
                        len(a),
                        len(b),
                        pair["wins"],
                        pair["losses"],
                        pair["p"],
                    ]
                )
