"""Verify the source text lengths, overlap and paired semantic-judgment scores."""

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from loopqwen.io import read


def words(text):
    return re.findall(r"\w+", text.casefold())


def grams(text):
    tokens = words(text)
    return {tuple(tokens[i : i + 5]) for i in range(len(tokens) - 4)}


def make(paper, results, result, finish):
    name = "meaning-alignment-01"
    spec = read(Path("research/experiments") / (name + ".json"))
    train, panel = (read(spec[k])["rows"] for k in ("training", "material"))
    texts = sorted({v[k] for v in train for k in ("left", "right")})
    ng = [grams(s) for s in texts]
    by = defaultdict(list)
    for i, g in enumerate(ng):
        for shingle in g:
            by[shingle].append(i)
    with (paper / "meaning-input-audit.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "anchor",
                "domain",
                "prompt_tokens",
                "side",
                "whitespace_words",
                "source_body_words",
                "nearest_training_fivegram_jaccard",
                "mentions_partner_post_id",
                "explicit_duplicate_marker",
            ]
        )
        for r in panel:
            for j, k in enumerate(("left", "right")):
                text = r[k]
                g = grams(text)
                hits = Counter(i for k5 in g for i in by[k5])
                closest = max(
                    (n / (len(g) + len(ng[i]) - n) for i, n in hits.items()), default=0
                )
                body = text.partition("\n")[-1] if r["domain"] == "AskUbuntu" else ""
                w.writerow(
                    [
                        r["id"],
                        r["pair"],
                        r["domain"],
                        len(r["prompt_ids"]),
                        k,
                        len(text.split()),
                        len(body.split()) if body else "",
                        closest,
                        r["posts"][1 - j] in text if "posts" in r else "",
                        bool(
                            re.search(
                                r"possible duplicate|marked as duplicate|exact duplicate",
                                text,
                                re.IGNORECASE,
                            )
                        ),
                    ]
                )
    if not (results / name / "checksums.json").is_file():
        return
    groups = result(name)["groups"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.35), sharey=True)
    with (paper / "meaning-outcomes.csv").open("w") as f:
        keys = list(groups["all"]["candidate"])
        w = csv.writer(f)
        w.writerow(["source", "model", *keys, "pair_gains", "pair_losses", "pair_p"])
        for i, group in enumerate(("all", "Wikipedia", "AskUbuntu")):
            record = groups[group]
            for side, offset, color, marker, label in (
                ("candidate", -0.12, "#11675d", "o", "Two passes"),
                ("control", 0.12, "#9d5932", "s", "Base"),
            ):
                m = record[side]
                w.writerow(
                    [
                        group,
                        side,
                        *[m[k] for k in keys],
                        record["gains"],
                        record["losses"],
                        record["p_exact"],
                    ]
                )
                for ax, k in zip(axes, ("pairs", "probe_pairs"), strict=True):
                    ax.scatter(
                        100 * m[k] / m["groups"],
                        i + offset,
                        color=color,
                        marker=marker,
                        s=30,
                        label=label if i == 0 else None,
                    )
    axes[0].set_yticks(range(3), ["All 40 anchors", "Wikipedia (24)", "AskUbuntu (16)"])
    axes[0].invert_yaxis()
    for ax, title in zip(
        axes,
        ("Two correct free answers (%)", "Two correct common-prefix probes (%)"),
        strict=True,
    ):
        ax.set_xlabel(title)
        ax.set_xlim(-4, 104)
        ax.set_xticks([0, 25, 50, 75, 100])
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=2, frameon=False)
    finish(fig, "meaning-pairs")
