"""Report the original prefix criterion and actual callback equivalence side by side."""

import csv

from loopqwen.evaluation.conversation_questions import score
from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read


def make(paper, results, result):
    rows = []
    for study in (
        "conversation-followup-01",
        "conversation-followup-modules-01",
        "conversation-followup-depth-01",
    ):
        path = results / study
        if not (path / "checksums.json").exists():
            continue
        result(study)
        checks = read(path / "checksums.json")
        material = read(read(path / "registration.json")["material"])
        for side in ("candidate", "control"):
            f = path / (side + ".json.gz")
            if digest(f) != checks[f.name]:
                raise ValueError("Changed conversation transcript")
            for r, w in zip(unpack(f), material["rows"], strict=True):
                literal = score(w, r["exchanges"], literal_prefix=True)
                operational = score(w, r["exchanges"])
                if r["score"] != (
                    literal if study == "conversation-followup-01" else operational
                ):
                    raise ValueError("Changed registered conversation score")
                rows.append(
                    [
                        study,
                        side,
                        r["id"],
                        w["kind"],
                        int(r["score"]["success"]),
                        int(literal["first"]),
                        int(operational["first"]),
                        int(operational["grounded"]),
                        int(operational["content"]),
                        int(operational["success"]),
                        r["wall_s"],
                        r["peak"],
                    ]
                )
    if rows:
        with (paper / "conversation-replay.csv").open("w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "study",
                    "side",
                    "id",
                    "family",
                    "registered_complete",
                    "literal_first",
                    "equivalent_first",
                    "observed_final_lookup",
                    "final_content",
                    "equivalent_complete",
                    "wall_s",
                    "peak_bytes",
                ]
            )
            writer.writerows(rows)
