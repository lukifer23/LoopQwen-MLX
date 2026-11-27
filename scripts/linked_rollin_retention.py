"""Compare both trained policies on the exact same retained user requests and Base outputs."""

import csv
from collections import defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read
from scripts.verify_runtime_wheel import trace


def compare():
    before, after = [
        Path("research/results") / n for n in ("valid-return-01", "linked-rollin-02")
    ]
    panels = [
        read(read(folder / "registration.json")["material"])
        for folder in (before, after)
    ]
    old = {w["id"]: w for w in panels[0]["rows"]}
    archive = [
        {
            arm: {
                w["id"]: w["requests"][0] for w in unpack(folder / (arm + ".json.gz"))
            }
            for arm in ("candidate", "control")
        }
        for folder in (before, after)
    ]
    details, families = [], defaultdict(list)
    for w in panels[1]["rows"]:
        key = w["id"]
        if key not in old:
            continue
        assert w == old[key]
        if w["document"]:
            assert (
                panels[0]["documents"][w["document"]]
                == panels[1]["documents"][w["document"]]
            )
        a, b = [v["control"][key] for v in archive]
        assert (
            trace(a["trajectory"]) == trace(b["trajectory"])
            and a["score"]["success"] == b["score"]["success"]
        )
        row = {
            "id": key,
            "family": w["family"],
            "base": a["score"]["success"],
            "previous_loop": archive[0]["candidate"][key]["score"]["success"],
            "new_loop": archive[1]["candidate"][key]["score"]["success"],
        }
        details.append(row)
        families[w["family"]].append(row)
    assert len(details) == 36
    with Path("paper/linked-rollin-retention.csv").open("w", newline="") as f:
        out = csv.DictWriter(f, details[0].keys(), lineterminator="\n")
        out.writeheader()
        out.writerows(details)
    groups = []
    for name, group in [*sorted(families.items()), ("All retained", details)]:
        gained = sum(r["new_loop"] and not r["previous_loop"] for r in group)
        lost = sum(r["previous_loop"] and not r["new_loop"] for r in group)
        groups.append(
            [
                name,
                str(len(group)),
                *[
                    str(sum(r[k] for r in group))
                    for k in ("previous_loop", "new_loop", "base")
                ],
                str(gained),
                str(lost),
            ]
        )
    return groups
