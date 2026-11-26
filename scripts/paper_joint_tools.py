"""Publish matched gains, losses and time on the four preregistered tool panels."""

import csv
from collections import Counter

import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read


def make(paper, results, result, finish):
    def records(name, side):
        result(name)
        path = results / name / (side + ".json.gz")
        if digest(path) != read(results / name / "checksums.json")[path.name]:
            raise ValueError("Different tool transcript")
        return {r["id"]: r for r in unpack(path)}

    name = "tool-dialogue-counterfactual-01"
    if not (results / name / "checksums.json").exists():
        return
    old = records("tool-dialogue-typed-01", "candidate") | records(
        "tool-dialogue-unified-01", "candidate"
    )
    new = records(name, "candidate")
    base = records(name, "control")
    items = read(read(results / name / "registration.json")["material"])["rows"]
    common = {
        w["id"]: w
        for w in read("research/data/tool-dialogue-unified-01.json")
        + read("research/data/tool-exec-bfcl-01.json")["rows"]
    }
    for w in items:
        if w["id"] in common:
            assert all(w[k] == v for k, v in common[w["id"]].items())
            assert (
                new[w["id"]]["trajectory"]["turns"][0]["prompt_ids"]
                == old[w["id"]]["trajectory"]["turns"][0]["prompt_ids"]
            )
    sources = [
        ("Original catalogs", "catalog"),
        ("Grounded catalogs", "grounded"),
        ("Public numerical", "public"),
        ("New domains", "fresh"),
    ]
    rows = []
    for label, panel in sources:
        ids = [w["id"] for w in items if w["panel"] == panel]

        def compare(arm):
            relevant = [k for k in ids if k in arm]
            deltas = Counter(
                int(new[k]["score"]["success"]) - int(arm[k]["score"]["success"])
                for k in relevant
            )
            return (
                (
                    sum(arm[k]["score"]["success"] for k in relevant),
                    deltas[1],
                    deltas[-1],
                )
                if relevant
                else ("", "", "")
            )

        rows.append(
            [
                label,
                len(ids),
                sum(new[k]["score"]["success"] for k in ids),
                *compare(base),
                *compare(old),
                sum(new[k]["trajectory"]["seconds"] for k in ids),
                sum(base[k]["trajectory"]["seconds"] for k in ids),
            ]
        )
    with (paper / "joint-tools.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "panel",
                "n",
                "joint_correct",
                "base_correct",
                "wins_over_base",
                "losses_to_base",
                "prior_correct",
                "wins_over_prior",
                "losses_to_prior",
                "joint_seconds",
                "base_seconds",
            ]
        )
        w.writerows(rows)
    fig, ax = plt.subplots(figsize=(6.5, 2.45))
    for i, (label, n, a, b, _, _, prior, _, _, _, _) in enumerate(rows):
        ax.plot([100 * b / n, 100 * a / n], [i, i], color="#cbd0cf", lw=2)
        ax.scatter(100 * b / n, i, c="#9d5932", marker="s", s=36)
        ax.scatter(100 * a / n, i - 0.055, c="#11675d", marker="o", s=40)
        if prior != "":
            ax.scatter(100 * prior / n, i + 0.055, c="#536a89", marker="^", s=43)
    ax.scatter([], [], c="#11675d", marker="o", label="New recurrent")
    ax.scatter([], [], c="#536a89", marker="^", label="Earlier recurrent")
    ax.scatter([], [], c="#9d5932", marker="s", label="Base")
    ax.legend(
        ncol=3,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.99),
        fontsize=8,
    )
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set(xlim=(-2, 102), xlabel="Correct complete tool sessions (%)")
    finish(fig, "joint-tools")


def matched_effects(paper, results, result, finish):
    """Paired cluster resampling, stratified by panel to retain its enrolled weight."""
    from collections import defaultdict

    import numpy as np

    study = "tool-dialogue-counterfactual-depth-01"
    joint = "tool-dialogue-counterfactual-01"
    if not (results / study / "checksums.json").is_file():
        return
    result(study)
    result(joint)

    def scores(name, arm):
        path = results / name / (arm + ".json.gz")
        if digest(path) != read(results / name / "checksums.json")[path.name]:
            raise ValueError("Changed paired transcripts")
        return {r["id"]: int(r["score"]["success"]) for r in unpack(path)}

    reference = scores(joint, "candidate")
    if reference != scores(study, "candidate"):
        raise ValueError("Matched candidate differs from the registered replay")
    comparisons = {
        "Untouched Base": scores(joint, "control"),
        "Equally trained K1": scores(study, "control"),
    }
    spec = read(results / joint / "registration.json")
    items = read(spec["material"])["rows"]
    panels = {
        "catalog": "Original catalogs",
        "grounded": "Grounded catalogs",
        "public": "Public numerical",
        "fresh": "New domains",
    }
    replicates = 20000
    rows = []
    for control, other in comparisons.items():
        if set(other) != set(reference) or set(other) != {r["id"] for r in items}:
            raise ValueError("Misspecified paired cohort")
        rng = np.random.default_rng(81572)
        pooled = np.zeros(replicates)
        for key, label in panels.items():
            groups = defaultdict(list)
            for r in items:
                if r["panel"] == key:
                    k = r["id"]
                    cluster = (
                        r["function"][0]["name"]
                        if key == "public"
                        else k.rsplit("-", 1)[0]
                    )
                    groups[cluster].append(reference[k] - other[k])
            blocks = [groups[k] for k in sorted(groups)]
            # Each world has five episodes. Each public function has two requests.
            if {len(b) for b in blocks} != {2 if key == "public" else 5}:
                raise ValueError("Unequal episode counts require a weighted bootstrap")
            n = sum(map(len, blocks))
            gains = np.array([sum(b) for b in blocks])
            draws = rng.choice(gains, size=(replicates, len(gains))).sum(axis=1)
            pooled += draws
            low, high = np.quantile(100 * draws / n, [0.025, 0.975])
            rows.append(
                (control, label, n, len(gains), 100 * gains.sum() / n, low, high)
            )
        n = len(items)
        low, high = np.quantile(100 * pooled / n, [0.025, 0.975])
        rows.append(
            (
                control,
                "All four panels",
                n,
                sum(r[3] for r in rows if r[0] == control),
                100 * (sum(reference.values()) - sum(other.values())) / n,
                low,
                high,
            )
        )
    with (paper / "matched-tool-effects.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "control",
                "panel",
                "items",
                "clusters",
                "difference_pp",
                "cluster_percentile_low_pp",
                "cluster_percentile_high_pp",
            ]
        )
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(6.1, 2.8))
    for j, (control, color) in enumerate(
        zip(comparisons, ("#9d5932", "#11675d"), strict=True)
    ):
        subset = [r for r in rows if r[0] == control]
        for i, (_, _, _, _, val, low, high) in enumerate(subset):
            y = i + (j - 0.5) * 0.26
            ax.hlines(y, low, high, color=color, lw=1.6)
            ax.scatter(
                val,
                y,
                s=22,
                c=color,
                marker=("s", "o")[j],
                label=control if not i else None,
            )
    ax.axvline(0, color="#68716f", linestyle=":", lw=1)
    ax.set_yticks(range(5), [*panels.values(), "All four panels"])
    ax.invert_yaxis()
    ax.set_xlabel("Two-pass model minus control (percentage points)")
    ax.legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    finish(fig, "matched-tool-effects")
