"""Describe source-level variation and the joint latency/quality of the same paired requests."""

import math
from collections import defaultdict

import numpy as np

from loopqwen.research.linked_rollin import exact_sign_flip


def paired_sources(rows):
    families = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["family"] == "open":
            continue
        families[r["family"]][r["cluster"]].append(r)
    output = []
    for family, groups in sorted(families.items()):
        counts = []
        changes = []
        up = down = 0
        for _, source in sorted(groups.items()):
            deltas = [
                int(r["candidate_grounded"]) - int(r["control_grounded"])
                for r in source
            ]
            counts.append(len(source))
            changes.append(sum(deltas))
            up += deltas.count(1)
            down += deltas.count(-1)
        n = sum(counts)
        # Resample complete registered sources. These are descriptive intervals conditional on
        # the measured checkpoint, tasks and sampling rules. Shared ecosystems can still correlate.
        rng = np.random.default_rng(41857)
        ix = rng.integers(len(groups), size=(10000, len(groups)))
        rates = np.array(changes)[ix].sum(axis=1) / np.array(counts)[ix].sum(axis=1)
        low, high = np.quantile(rates, [0.025, 0.975]).tolist()
        output.append(
            {
                "family": family,
                "requests": n,
                "registered_groups": len(groups),
                "gains": up,
                "losses": down,
                "net_rate": (up - down) / n,
                "resampled_low": low,
                "resampled_high": high,
                "exact_group_p": exact_sign_flip(changes),
            }
        )
    # Holm step-down controls the familywise error under arbitrary cross-family dependencies
    # when the individual group exchangeability assumptions hold.
    ordered = sorted(output, key=lambda r: r["exact_group_p"])
    prior = 0.0
    for j, r in enumerate(ordered):
        prior = min(1.0, max(prior, (len(ordered) - j) * r["exact_group_p"]))
        r["holm_p"] = prior
    return output


def cost_profile(rows):
    records = []
    for measure, factor in (
        ("wall_s", 1.0),
        ("first_s", 1.0),
        ("peak_bytes", 1 / 2**20),
    ):
        left, right = (
            np.array([r[a + "_" + measure] * factor for r in rows])
            for a in ("candidate", "control")
        )
        delta = left - right
        records.append(
            {
                "metric": measure if measure != "peak_bytes" else "peak_MiB",
                "requests": len(rows),
                "loop_lower": int((delta < 0).sum()),
                "equal": int((delta == 0).sum()),
                "base_lower": int((delta > 0).sum()),
                **{
                    a + "_p" + str(p): float(np.percentile(v, p))
                    for a, v in (
                        ("loop", left),
                        ("base", right),
                        ("paired_delta", delta),
                    )
                    for p in (50, 90, 95)
                },
            }
        )
    return records


def joint(rows):
    numeric = [r for r in rows if r["family"] != "open"]
    qualified = [
        r for r in numeric if r["candidate_grounded"] and not r["control_grounded"]
    ]
    return {
        "net_new_cases": len(qualified),
        "additional_correct_and_all_three_costs_lower": sum(
            all(
                r["candidate_" + m] < r["control_" + m]
                for m in ("wall_s", "first_s", "peak_bytes")
            )
            for r in qualified
        ),
        "both_correct": sum(
            r["candidate_grounded"] and r["control_grounded"] for r in numeric
        ),
        "neither_correct": sum(
            not r["candidate_grounded"] and not r["control_grounded"] for r in numeric
        ),
    }


def open_p(a, b):
    wins = [z.get("win", 0) for z in (a, b)]
    total = sum(wins)
    return min(
        1.0, 2 * sum(math.comb(total, k) for k in range(min(wins) + 1)) / 2**total
    )
