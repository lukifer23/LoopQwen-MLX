"""Compare paired supported generations within frozen source, world and deadline."""

import argparse
from collections import defaultdict
from statistics import fmean

from loopqwen.io import read, save
from scripts.middle_depth_budget import ROOT


def mechanism_item(family, world):
    return (
        family == "reasoning"
        or (family == "tools" and world in ("original", "changed", "irrelevant"))
        or (family == "chat" and world == "changed")
    )


def result(stage, candidate, control, cap):
    rows = read(ROOT / "review" / stage / "scored.json")

    def arm(a):
        return {
            (r["family"], r["source"], r["world"]): r
            for r in rows
            if r["arm"] == a and r["cap"] == cap
        }

    a, b = arm(candidate), arm(control)
    assert a and a.keys() == b.keys()
    paired = []
    for key in sorted(a):
        x, y = a[key], b[key]
        if key[2] != "own" and x["input_tokens"] != y["input_tokens"]:
            raise ValueError("Paired arm opening lengths differ")
        ax, by = x["grade"]["correct"], y["grade"]["correct"]
        paired.append(
            dict(
                family=key[0],
                source=key[1],
                world=key[2],
                gain=int(ax) - int(by),
                left=x,
                right=y,
            )
        )
    families = defaultdict(lambda: defaultdict(list))
    for row in paired:
        if row["world"] in ("original", "changed", "irrelevant"):
            families[row["family"]][row["source"]].append(row)

    def rate(v, which, deadline):
        r = v[which]
        return int(
            r["grade"]["correct"] and (deadline is None or r["seconds"] <= deadline)
        )

    curves = []
    for deadline in (2, 4, 8, 16, 32, 64, None):
        record = {}
        for family, sources in families.items():
            record[family] = {
                which: fmean(
                    fmean(rate(r, which, deadline) for r in worlds)
                    for worlds in sources.values()
                )
                for which in ("left", "right")
            }
        curves.append(
            dict(
                seconds=deadline,
                families=record,
                macro=fmean(z["left"] - z["right"] for z in record.values()),
            )
        )
    mechanisms = [r for r in paired if mechanism_item(r["family"], r["world"])]
    regroup = defaultdict(list)
    for r in mechanisms:
        regroup[r["source"]].append(r)
    effect = fmean(
        fmean(rate(r, "left", 8) - rate(r, "right", 8) for r in rs)
        for rs in regroup.values()
    )
    save(
        ROOT / "pairs" / f"{stage}-{candidate}-{control}-{cap}.json",
        dict(
            stage=stage,
            cap=cap,
            candidate=candidate,
            control=control,
            trajectories=paired,
            curves=curves,
            mechanism_by_deadline=effect,
            gains=sum(r["gain"] == 1 for r in paired),
            losses=sum(r["gain"] == -1 for r in paired),
            control_successes=sum(r["right"]["grade"]["correct"] for r in paired),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage")
    p.add_argument("candidate")
    p.add_argument("control")
    p.add_argument("cap", type=int)
    a = p.parse_args()
    result(a.stage, a.candidate, a.control, a.cap)
