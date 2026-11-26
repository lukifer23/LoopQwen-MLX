"""Paired document-level uncertainty for the raw language comparisons."""

import csv
import random

import matplotlib.pyplot as plt
from loopqwen.evaluation.mechanism import unpack


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
