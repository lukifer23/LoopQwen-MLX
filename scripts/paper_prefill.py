"""Relate measured full-history prefill time to the number of causal key/query pairs."""

import csv
from collections import defaultdict
from statistics import mean

import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read


def make(paper, results, result, finish):
    rows = []
    series = {
        "pages": ("Recurrent, selected pages", "#11675d", "o"),
        "stock": ("Recurrent, dense", "#536a89", "^"),
        "base": ("Untouched Base", "#9d5932", "s"),
    }
    for study, control in (
        ("register-page-prefill-01", "stock"),
        ("register-page-full-01", "base"),
    ):
        path = results / study
        if not (path / "checksums.json").is_file():
            continue
        result(study)
        checks = read(path / "checksums.json")
        for side in ("pages", control):
            f = path / (side + ".json.gz")
            if digest(f) != checks[f.name]:
                raise ValueError("Changed prefill cohort")
            for r in unpack(f):
                sites = r["kv_history"]
                consumed = len(r["token_ids"]) + r["prompt_tokens"] - 1
                if not all(s["retained"] == s["offset"] == consumed for s in sites):
                    raise ValueError("Incomplete physical history")
                possible = len(sites) * consumed * (consumed + 1) // 2
                seen = (
                    sum(s["attended"] for s in r["page_sites"])
                    if side == "pages"
                    else possible
                )
                rows.append(
                    (
                        study,
                        side,
                        r["prompt_tokens"],
                        len(r["token_ids"]),
                        int(r["correct"]),
                        r["ttft_s"],
                        r["elapsed_s"],
                        r["peak_bytes"],
                        len(sites),
                        consumed,
                        seen,
                        possible,
                    )
                )
    if not rows:
        return
    with (paper / "page-prefill.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "study",
                "side",
                "prompt_tokens",
                "output_tokens",
                "correct",
                "ttft_s",
                "elapsed_s",
                "peak_bytes",
                "global_sites",
                "physically_retained_per_site",
                "causal_pairs_attended",
                "causal_pairs_possible",
            ]
        )
        w.writerows(rows)
    fig, (timing, work) = plt.subplots(1, 2, figsize=(6.65, 2.65))
    xvalues = sorted({r[2] for r in rows})
    for name, (label, color, marker) in series.items():
        groups = defaultdict(list)
        for r in rows:
            if r[1] == name:
                groups[r[2]].append(r)
        if not groups:
            continue
        xs = sorted(groups)
        timing.plot(
            xs,
            [mean(r[5] for r in groups[x]) for x in xs],
            color=color,
            marker=marker,
            label=label,
            markersize=5,
            linewidth=1.3,
        )
        if name == "pages":
            work.bar(
                [str(round(x / 1024)) + "K" for x in xs],
                [
                    100 * sum(r[-2] for r in groups[x]) / sum(r[-1] for r in groups[x])
                    for x in xs
                ],
                color=color,
                width=0.52,
            )
    timing.set_xscale("log", base=2)
    timing.set_xticks(xvalues, [str(round(x / 1024)) + "K" for x in xvalues])
    if len(xvalues) == 2:
        timing.set_xlim(xvalues[0] * 0.90, xvalues[-1] * 1.10)
    timing.set_xlabel("Input tokens")
    timing.set_ylabel("Mean first-token time (s)")
    timing.legend(fontsize=7, frameon=False)
    work.set_xlabel("Input tokens")
    work.set_ylabel("Causal pairs attended (%)")
    work.set_ylim(0, 105)
    work.axhline(100, c="#536a89", lw=1, ls=":")
    finish(fig, "page-prefill")


def resources(paper, results, result, finish):
    comparisons = (
        (
            "register-page-prefill-01",
            "pages",
            "stock",
            "BF16 pages vs dense recurrent · 16K/32K",
        ),
        ("register-page-full-01", "pages", "base", "BF16 pages vs Base · 131K"),
        (
            "register-packed-pilot-01",
            "packed_pages",
            "pages",
            "8-bit vs BF16 pages · 16K/32K",
        ),
        (
            "register-packed-direct-pilot-01",
            "direct_pages",
            "pages",
            "Direct 8-bit vs BF16 pages · 16K/32K",
        ),
        (
            "register-packed-direct-full-01",
            "direct_pages",
            "base",
            "Direct 8-bit pages vs Base · 131K",
        ),
    )
    panels = []
    with (paper / "prefill-resource-comparisons.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "study",
                "metric",
                "candidate",
                "control",
                "unit",
                "candidate_correct",
                "control_correct",
                "requests",
                "passed",
            ]
        )
        for name, first, second, title in comparisons:
            if not (results / name / "checksums.json").is_file():
                continue
            d = result(name)
            a, b = (d["metrics"][key] for key in (first, second))
            changes = []
            for key, unit in (
                ("ttft", "seconds"),
                ("seconds", "seconds"),
                ("peak", "bytes"),
            ):
                writer.writerow(
                    [
                        name,
                        key,
                        a[key],
                        b[key],
                        unit,
                        a["correct"],
                        b["correct"],
                        d["items"],
                        d["passed"],
                    ]
                )
                changes.append(100 * (a[key] / b[key] - 1))
            panels.append((title, changes))
    if not panels:
        return
    fig, ax = plt.subplots(figsize=(6.65, max(1.95, 0.43 * len(panels) + 1.2)))
    for i, (title, changes) in enumerate(panels):
        for j, (value, color, marker, metric) in enumerate(
            zip(
                changes,
                ("#11675d", "#4773a1", "#ad5d33"),
                ("o", "^", "s"),
                ("First token", "Total time", "Peak bytes"),
                strict=True,
            )
        ):
            ax.scatter(
                value,
                i + (j - 1) * 0.14,
                c=color,
                marker=marker,
                s=30,
                label=metric if i == 0 else None,
            )
    ax.axvline(0, color="#85918d", linewidth=0.85)
    ax.set_yticks(range(len(panels)), [t for t, _ in panels], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Change from the paired control (%)")
    ax.legend(
        frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1), fontsize=8
    )
    finish(fig, "prefill-resource-comparisons")
