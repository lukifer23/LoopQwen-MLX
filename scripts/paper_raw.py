"""Paired document-level uncertainty for the raw language comparisons."""

import csv
import random

import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read


def corpus_intervals(rows, seed):
    rng = random.Random(seed)
    n = len(rows)

    # Resample whole texts. The statistic weights their losses by target-token count.
    def estimate(ds):
        return sum(a for a, _ in ds) / sum(b for _, b in ds)

    boot = sorted(estimate(rng.choices(rows, k=n)) for _ in range(2000))
    return estimate(rows), boot[49], boot[1949]


def make(paper, results, result, finish):
    labels = {
        "cache-language-01": "Causal token memory",
        "raw-distillation-01": "Larger-teacher distillation",
    }
    domains = [
        ("wikitext", "WikiText"),
        ("general_prose", "General prose"),
        ("scientific_exposition", "Scientific exposition"),
        ("reference", "Reference"),
        ("technical_explanation", "Technical explanation"),
        ("code_description", "Code description"),
    ]
    studies = [
        (name, label)
        for name, label in labels.items()
        if (results / name / "checksums.json").is_file()
    ]
    if not studies:
        return
    fig, axes = plt.subplots(
        1,
        len(studies),
        figsize=(3.3 * len(studies) + 2.2, 3.0),
        squeeze=False,
        sharey=True,
    )
    with (paper / "raw-language.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "study",
                "domain",
                "documents",
                "targets",
                "candidate_nll",
                "base_nll",
                "difference",
                "paired_bootstrap_lower",
                "paired_bootstrap_upper",
                "candidate_top1",
                "base_top1",
                "candidate_seconds",
                "base_seconds",
            ]
        )
        for j, (name, label) in enumerate(studies):
            scores = result(name)["scores"]
            ax = axes[0, j]
            data = [
                unpack(results / name / (s + ".json.gz"))
                for s in ("candidate", "control")
            ]
            for i, (domain, _) in enumerate(domains):
                ds = []
                for a, b in zip(*data, strict=True):
                    if a["category"] != domain:
                        continue
                    assert a["id"] == b["id"] and len(a["nll"]) == len(b["nll"])
                    ds.append((sum(a["nll"]) - sum(b["nll"]), len(a["nll"])))
                diff, lo, hi = corpus_intervals(ds, 17037 + i)
                c, b = [scores[s][domain] for s in ("candidate", "control")]
                writer.writerow(
                    [
                        name,
                        domain,
                        len(ds),
                        c["tokens"],
                        c["nll"],
                        b["nll"],
                        diff,
                        lo,
                        hi,
                        c["top1"],
                        b["top1"],
                        c["seconds"],
                        b["seconds"],
                    ]
                )
                ax.plot([lo, hi], [i, i], color="#5a7772", lw=1.3)
                ax.scatter(diff, i, color="#11675d", s=32, zorder=3)
            ax.axvline(0, color="#8e9594", ls="--", lw=0.9)
            ax.set_title(label)
            ax.set_xlabel("NLL difference (nats/token)")
            ax.set_yticks(range(len(domains)), [v for _, v in domains])
            if j == 0:
                ax.invert_yaxis()
    finish(fig, "raw-language")


def selective(paper, results, result, finish):
    """Expose source-level changes on the genuinely new passages."""
    from collections import defaultdict

    name = "selective-raw-01"
    root = results / name
    if not (root / "checksums.json").exists():
        return
    d = result(name)
    spec = read(root / "registration.json")
    new = read(spec["material"])["rows"]
    a, b = (
        unpack(root / (side + ".json.gz"))[: len(new)]
        for side in ("candidate", "control")
    )
    sources = defaultdict(list)
    for u, v, w in zip(a, b, new, strict=True):
        assert u["id"] == v["id"] == w["id"]
        gold = w["ids"][1:]
        change = sum(
            int(x == z) - int(y == z)
            for x, y, z in zip(u["ids"], v["ids"], gold, strict=True)
        )
        sources[w["source"]].append((sum(u["nll"]) - sum(v["nll"]), change, len(gold)))
    names = {"pg215": "The Call of the Wild", "pg944": "The Voyage of the Beagle"}
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.2, 3.25), sharey=True)
    with (paper / "selective-sources.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source",
                "passages",
                "targets",
                "nll_difference",
                "net_correct",
                "net_correct_per_1000",
            ]
        )
        for i, (source, rs) in enumerate(sources.items()):
            nll, top, total = map(sum, zip(*rs, strict=True))
            writer.writerow(
                [source, len(rs), total, nll / total, top, 1000 * top / total]
            )
            left.scatter(nll / total, i, color="#11675d")
            right.scatter(1000 * top / total, i, color="#11675d")
    for ax in (left, right):
        ax.axvline(0, color="#777", lw=0.8)
    left.set_yticks(
        range(len(sources)), [names.get(s, s.removeprefix("library/")) for s in sources]
    )
    left.invert_yaxis()
    left.set_xlabel("NLL difference (nats/token)")
    right.set_xlabel("Additional correct per 1,000 tokens")
    finish(fig, "selective-sources")
    with (paper / "selective-domains.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["domain", "side", "tokens", "nll", "top1", "seconds", "peak_bytes"]
        )
        for side, groups in d["scores"].items():
            for domain, vals in groups.items():
                writer.writerow(
                    [
                        domain,
                        side,
                        *[
                            vals[k]
                            for k in ("tokens", "nll", "top1", "seconds", "peak")
                        ],
                    ]
                )


def future_sources(paper, results, result, finish):
    name = "future-token-prose-01"
    root = results / name
    if not (root / "checksums.json").exists():
        return
    d = result(name)
    spec = read(root / "registration.json")
    rows = read(spec["material"])["rows"]
    categories = {r["source"]: r["category"] for r in rows}
    names = ("fiction", "memoir", "manual", "encyclopedia")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), sharey=True)
    with (paper / "future-token-sources.csv").open("w") as stream:
        out = csv.writer(stream)
        out.writerow(
            [
                "source",
                "domain",
                "passages",
                "tokens",
                "nll_change_per_token",
                "additional_correct_top1",
            ]
        )
        for y, domain in enumerate(names):
            sources = sorted(s for s, c in categories.items() if c == domain)
            for j, source in enumerate(sources):
                nll, change, n = d["source_totals"][source]
                out.writerow(
                    [
                        source,
                        domain,
                        sum(r["source"] == source for r in rows),
                        n,
                        nll / n,
                        change,
                    ]
                )
                shift = (j - (len(sources) - 1) / 2) * 0.065
                axes[0].scatter(nll / n, y + shift, color="#11675d", s=17)
                axes[1].scatter(1000 * change / n, y + shift, color="#536a89", s=17)
    for ax in axes:
        ax.axvline(0, color="#888", linewidth=0.8)
    axes[0].set_yticks(range(4), ["Fiction", "Memoir", "Git manuals", "Encyclopedia"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("NLL change (nats/token)")
    axes[1].set_xlabel("Net correct per 1,000 tokens")
    finish(fig, "future-token-sources")
