"""Relate measured full-history prefill time to the number of causal key/query pairs."""

import csv
from collections import defaultdict
from statistics import mean, median

import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read


def decode_operator(paper, results, result, finish, name="packed-decode-kernel-01"):
    original = name == "packed-decode-kernel-01"
    target = "packed-decode" if original else "factored-decode"
    arms = (
        (("stock", "MLX-LM", "#536a89", "s"), ("fused", "Fused Metal", "#11675d", "o"))
        if original
        else (
            ("fused", "Prior fused", "#536a89", "s"),
            ("factored", "Factored affine", "#11675d", "o"),
        )
    )
    root = results / name
    if not (root / "checksums.json").is_file():
        return
    decision = result(name)
    times = root / "events.json"
    if digest(times) != read(root / "checksums.json")[times.name]:
        raise ValueError("Changed decoder latency observations")
    events = read(times)
    lengths = read(root / "registration.json")["lengths"]
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.5), sharey=True)
    with (paper / (target + ".csv")).open("w") as file:
        w = csv.writer(file)
        w.writerow(
            [
                "physical_length",
                "mask",
                "operator",
                "calls",
                "median_ms",
                "max_extra_bytes",
            ]
        )
        for ax, mask in zip(axes, ("plain", "reader"), strict=True):
            for side, label, color, marker in arms:
                medians = []
                for pos, n in enumerate(lengths):
                    rows = [
                        e
                        for e in events
                        if (e["length"], e["mask"], e["side"]) == (n, mask, side)
                    ]
                    group = decision["groups"][f"{n}-{mask}"]
                    seconds = median(e["seconds"] for e in rows)
                    peak = max(e["extra_bytes"] for e in rows)
                    if (
                        seconds != group["median_seconds"][side]
                        or peak != group["extra_peak_bytes"][side]
                    ):
                        raise ValueError(
                            "Kernel timing differs from its archived decision"
                        )
                    medians.append(1000 * seconds)
                    w.writerow([n, mask, side, len(rows), medians[-1], peak])
                    # Show one median per actual input query, retaining its four timed repeats.
                    for prefix in sorted({e["prefix"] for e in rows}):
                        val = median(
                            e["seconds"] for e in rows if e["prefix"] == prefix
                        )
                        ax.scatter(pos, 1000 * val, color=color, alpha=0.35, s=12)
                ax.plot(
                    range(len(lengths)),
                    medians,
                    color=color,
                    marker=marker,
                    label=label,
                )
            ax.set_xticks(range(len(lengths)), [f"{n:,}" for n in lengths])
            ax.set_xlabel("Physical KV positions")
            ax.set_title(
                "No bias" if mask == "plain" else "Learned reader bias", fontsize=9
            )
            ax.set_yscale("log")
        axes[0].set_ylabel("Attention call (ms, log scale)")
        axes[1].legend(frameon=False, fontsize=8)
    finish(fig, target)


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
        (
            "register-fused-base-full-01",
            "direct_pages",
            "base",
            "Fused 8-bit pages vs Base · fresh 131K",
        ),
    )
    with (paper / "long-throughput.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "study",
                "arm",
                "requests",
                "input_tokens",
                "output_tokens",
                "post_first_tokens",
                "ttft_seconds",
                "decode_seconds",
                "prompt_tokens_per_first_token_second",
                "post_first_tokens_per_second",
            ]
        )
        for name, first, second, _ in comparisons:
            root = results / name
            if not (root / "checksums.json").is_file():
                continue
            for arm in (first, second):
                file = root / (arm + ".json.gz")
                if digest(file) != read(root / "checksums.json")[file.name]:
                    raise ValueError("Changed long-context measurements")
                rows = unpack(file)
                n = sum(r["prompt_tokens"] for r in rows)
                outputs = sum(len(r["token_ids"]) for r in rows)
                prefill = sum(r["ttft_s"] for r in rows)
                decode = sum(r["elapsed_s"] - r["ttft_s"] for r in rows)
                steps = outputs - len(rows)
                w.writerow(
                    [
                        name,
                        arm,
                        len(rows),
                        n,
                        outputs,
                        steps,
                        prefill,
                        decode,
                        n / prefill,
                        steps / decode,
                    ]
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


def full_decoder(paper, results, result, finish):
    name = "packed-decode-full-01"
    root = results / name
    if not (root / "checksums.json").exists():
        return
    d = result(name)
    f = root / "runs.json.gz"
    if digest(f) != read(root / "checksums.json")[f.name]:
        raise ValueError("Full decoder run inventory changed")
    records = unpack(f)
    fig, (prefill, decode) = plt.subplots(1, 2, figsize=(7.3, 2.65))
    with (paper / "full-decoder.csv").open("w") as out:
        w = csv.writer(out)
        w.writerow(
            [
                "sequence",
                "implementation",
                "document",
                "repeat",
                "correct",
                "prompt_tokens",
                "output_tokens",
                "decode_steps",
                "first_token_s",
                "total_s",
                "post_first_s",
                "post_first_tps",
                "peak_bytes",
                "fused_site_calls",
                "attention_sites",
            ]
        )
        seen = set()
        for rec in records:
            r = rec["result"]
            side = rec["side"]
            item = rec["item"]
            seq = rec["sequence"]
            steps = len(r["token_ids"]) - 1
            after = r["elapsed_s"] - r["ttft_s"]
            calls = sum(k.get("fused_queries", 0) for k in r["page_sites"])
            w.writerow(
                [
                    seq + 1,
                    side,
                    item + 1,
                    rec["repeat"] + 1,
                    r["correct"],
                    r["prompt_tokens"],
                    steps + 1,
                    steps,
                    r["ttft_s"],
                    r["elapsed_s"],
                    after,
                    steps / after,
                    r["peak_bytes"],
                    calls,
                    len(r["kv_history"]),
                ]
            )
            color, marker, label = (
                ("#11675d", "o", "Fused output")
                if side == "fused"
                else ("#9d5932", "s", "Stock output")
            )
            for ax, y in [(prefill, r["ttft_s"]), (decode, steps / after)]:
                ax.scatter(
                    seq + 1,
                    y,
                    color=color,
                    marker=marker,
                    s=35,
                    label=label if side not in seen else None,
                )
            seen.add(side)
    for ax, title in (
        (prefill, "Seconds to first token"),
        (decode, "Post-first tokens per second"),
    ):
        ax.set_ylabel(title)
        ax.set_xlabel("Full request in execution order")
        ax.set_xticks(range(1, len(records) + 1))
        ax.set_ylim(bottom=0)
    prefill.legend(
        frameon=False, loc="lower left", bbox_to_anchor=(-0.1, 1), ncol=2, fontsize=8
    )
    assert len(records) == 2 * d["metrics"]["fused"]["queries"]
    finish(fig, "full-decoder")


def direct_fused(
    paper,
    results,
    result,
    finish,
    *,
    study="register-fused-base-full-01",
    prefix="fused-base",
):
    root = results / study
    if not (root / "checksums.json").is_file():
        return
    rates = result(study)["throughput"]
    sums = read(root / "checksums.json")
    mixed = read(root / "registration.json").get("mixed_register", False)
    with (paper / (prefix + "-requests.csv")).open("w") as stream:
        w = csv.writer(stream)
        w.writerow(
            [
                "side",
                "swap",
                "input_sha256",
                "prompt_tokens",
                "output_tokens",
                "answer",
                "expected",
                "correct",
                "first_s",
                "total_s",
                "post_first_s",
                "post_first_tps",
                "prefill_tps",
                "peak_bytes",
                "history_sites",
                "min_history_tokens",
                "fused_calls",
            ]
            + (
                [
                    "conversion_s",
                    "conversion_peak_bytes",
                    "new_tensor_bytes",
                    "converted_feedforward_maps",
                    "actual_output_projections",
                ]
                if mixed
                else []
            )
        )
        for side in ("direct_pages", "base"):
            path = root / (side + ".json.gz")
            assert digest(path) == sums[path.name]
            for j, r in enumerate(unpack(path), 1):
                precision = r.get("weight_precision", {})
                w.writerow(
                    [
                        side,
                        j,
                        r["input_sha256"],
                        r["prompt_tokens"],
                        len(r["token_ids"]),
                        r["answer"],
                        r["gold"],
                        r["correct"],
                        r["ttft_s"],
                        r["elapsed_s"],
                        r["elapsed_s"] - r["ttft_s"],
                        r["decode_tps"],
                        r["prefill_tps"],
                        r["peak_bytes"],
                        len(r["kv_history"]),
                        min(h["retained"] for h in r["kv_history"]),
                        sum(h.get("fused_queries", 0) for h in r.get("page_sites", [])),
                    ]
                    + (
                        [
                            precision.get("setup_s"),
                            precision.get("setup_peak"),
                            precision.get("new_resident_tensor_bytes"),
                            len(precision["projections"]) if precision else None,
                            precision.get("actual_projections"),
                        ]
                        if mixed
                        else []
                    )
                )
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.0))
    for ax, key, label in zip(
        axes,
        ("prefill_tps", "decode_tps", "peak"),
        ("Prompt tokens/s", "Post-first tokens/s", "Peak MLX (GiB)"),
        strict=True,
    ):
        for i, (side, c, title) in enumerate(
            (("direct_pages", "#146558", "Recurrent"), ("base", "#99552d", "Base"))
        ):
            y = rates[side][key] / (2**30 if key == "peak" else 1)
            ax.barh(i, y, color=c, height=0.55)
            ax.text(
                y,
                i,
                f" {y:,.2f}" if key != "prefill_tps" else f" {y:,.0f}",
                va="center",
                fontsize=8,
            )
        ax.set_yticks([0, 1], ["Recurrent", "Base"] if ax is axes[0] else ["", ""])
        ax.invert_yaxis()
        ax.set_xlim(0, ax.get_xlim()[1] * 1.35)
        ax.set_xlabel(label)
    finish(fig, prefix + "-full")
