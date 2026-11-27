"""Measure item-level stability and repeat grading in two archived comparisons."""

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typst

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.native_function_study import executions
from scripts.assistance_deliberation import write_csv
from scripts.assistance_uncertainty import paired_sources
from scripts.table_policy_reading import committed

PAPER = Path("paper")
DOC = Path("docs/measurements")
RESULT = Path("research/results")
STAGES = ("table-policy-gradient-01", "conditional-program-primary-01")
RATINGS = ("absent", "wrong", "partial", "fully usable")


def masked(program, reasoning):
    mask, grades = {}, {}
    for prefix in (program, reasoning):
        a, b = (DOC / (prefix + "-blinded" + ext + ".json") for ext in ("", "-ratings"))
        for p in (a, b):
            committed(p)
        m, g = read(a), read(b)
        assert m.keys() == g.keys() and not mask.keys() & m.keys()
        mask.update(m)
        grades.update(g)
    return mask, grades


def stable(rows):
    counts = Counter((r["earlier"], r["current"], r["base"]) for r in rows)
    return {
        "requests": len(rows),
        "both_gain": counts[1, 1, 0],
        "earlier_only_gain": counts[1, 0, 0],
        "current_only_gain": counts[0, 1, 0],
        "neither_correct": counts[0, 0, 0],
        "both_loss": counts[0, 0, 1],
        "earlier_only_loss": counts[0, 1, 1],
        "current_only_loss": counts[1, 0, 1],
        "all_correct": counts[1, 1, 1],
    }


def agreement(rows):
    pairs = Counter((r["earlier_rating"], r["current_rating"]) for r in rows)
    n = len(rows)
    equal = sum(pairs[v, v] for v in RATINGS)
    left = Counter(r["earlier_rating"] for r in rows)
    right = Counter(r["current_rating"] for r in rows)
    chance = sum(left[v] * right[v] for v in RATINGS) / n**2
    return {
        "n": n,
        "exact": equal,
        "kappa": round((equal / n - chance) / (1 - chance), 3) if chance < 1 else None,
        "usable_equal": sum(
            (a == "fully usable") == (b == "fully usable")
            for a, b in ((r["earlier_rating"], r["current_rating"]) for r in rows)
        ),
        "unsupported_equal": sum(
            r["earlier_unsupported"] == r["current_unsupported"] for r in rows
        ),
        "earlier_usable": left["fully usable"],
        "current_usable": right["fully usable"],
        "matrix": [[pairs[a, b] for b in RATINGS] for a in RATINGS],
    }


def build():
    plt.rcParams.update({"font.size": 8, "svg.hashsalt": "loopqwen-stability"})
    oldmask, oldgrade = masked("table-policy", "table-reasoning")
    newmask, newgrade = masked("conditional-program", "conditional-reasoning")
    panel = {x["id"]: x for x in read("research/data/table-policy-panel.json")["rows"]}
    assert len(panel) == 268 and len(oldmask) == 132
    objective, readings = [], []
    inputs = {}
    streams = {}
    for stage in STAGES:
        streams[stage] = {}
        for arm in ("candidate", "control"):
            path = RESULT / stage / (arm + ".json.gz")
            inputs[str(path)] = digest(path)
            streams[stage][arm] = {r["id"]: r["requests"][0] for r in unpack(path)}
    for key, item in panel.items():
        loop, base = (
            [streams[s][arm][key] for s in STAGES] for arm in ("candidate", "control")
        )
        for q in loop + base:
            assert q["trajectory"]["turns"][0]["prompt_ids"] == item["initial_ids"]
        old, new = (q["trajectory"] for q in base)
        assert [t["token_ids"] for t in old["turns"]] == [
            t["token_ids"] for t in new["turns"]
        ]
        assert (
            executions(old) == executions(new) and base[0]["score"] == base[1]["score"]
        )
        if item["family"] != "open":
            objective.append(
                {
                    "id": key,
                    "family": item["family"],
                    "cluster": item.get("cluster", item.get("source", key)),
                    "earlier": loop[0]["score"]["success"],
                    "current": loop[1]["score"]["success"],
                    "base": base[0]["score"]["success"],
                }
            )
        if key not in oldmask:
            continue
        for field in ("family", "request", "reference"):
            assert oldmask[key][field] == newmask[key][field]
        for arm, pair in (("control", base), ("candidate", loop)):
            texts = [q["trajectory"]["answer"] or "" for q in pair]
            if texts[0] != texts[1]:
                assert arm == "candidate"
                continue
            earlier, current = (
                g[key]["replies"][m[key]["replies"].index(texts[0])]
                for m, g in ((oldmask, oldgrade), (newmask, newgrade))
            )
            readings.append(
                {
                    "id": key,
                    "family": item["family"],
                    "arm": arm,
                    "text_hash": fingerprint(texts[0]),
                    "nonempty": bool(texts[0].strip()),
                    "earlier_rating": earlier["rating"],
                    "current_rating": current["rating"],
                    "earlier_unsupported": earlier["unsupported"],
                    "current_unsupported": current["unsupported"],
                }
            )
    assert len(objective) == 256
    families = sorted({r["family"] for r in objective})
    by_family = {
        f: stable([r for r in objective if r["family"] == f]) for f in families
    }
    intervals = paired_sources(
        [
            dict(r, candidate_grounded=r["current"], control_grounded=r["earlier"])
            for r in objective
        ]
    )
    groups = {}
    for arm in ("control", "candidate"):
        selected = [r for r in readings if r["arm"] == arm]
        for label, sample in (
            ("all", selected),
            ("nonempty", [r for r in selected if r["nonempty"]]),
        ):
            groups[arm + "_" + label] = agreement(sample)
    assert groups["control_all"]["n"] == 132
    write_csv(PAPER / "conditional-stability-objective.csv", objective)
    write_csv(PAPER / "conditional-stability-ratings.csv", readings)
    save(
        PAPER / "conditional-stability.json",
        {
            "streams": inputs,
            "all": stable(objective),
            "by_family": by_family,
            "source_intervals": intervals,
            "repeat_review": groups,
        },
    )
    fig, (a, b) = plt.subplots(
        1,
        2,
        figsize=(9, 3.6),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.35, 1]},
    )
    for i, d in enumerate(intervals):
        a.plot(
            [100 * d["resampled_low"], 100 * d["resampled_high"]],
            [i, i],
            color="#52717d",
        )
        a.scatter(100 * d["net_rate"], i, c="#a9452d", s=22)
    a.axvline(0, color="#9c9c9c", lw=0.8)
    a.set_yticks(
        range(len(intervals)), [r["family"].replace("_", " ") for r in intervals]
    )
    a.set_xlabel("Current minus earlier loop, percentage points")
    matrix = groups["control_nonempty"]["matrix"]
    b.imshow(matrix, cmap="Blues", vmin=0)
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            b.text(
                j,
                i,
                str(v),
                ha="center",
                va="center",
                color="white" if v > max(map(max, matrix)) * 0.55 else "#182329",
            )
    labels = ["Absent", "Wrong", "Partial", "Full"]
    b.set_xticks(range(4), labels)
    b.set_yticks(range(4), labels)
    b.set_xlabel("Second reading")
    b.set_ylabel("First reading")
    fig.savefig(PAPER / "conditional-stability.svg", metadata={"Date": None})
    plt.close(fig)
    typst.compile(
        str(PAPER / "conditional-stability.typ"),
        output=str(PAPER / "conditional-stability.pdf"),
        root=".",
        timestamp=0,
    )


if __name__ == "__main__":
    build()
