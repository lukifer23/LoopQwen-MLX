"""Attribute observed net gains to common versus one-sided completions across direct studies."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest
from scripts.assistance_deliberation import write_csv
from scripts.assistance_uncertainty import completed_pairs

STUDIES = (
    ("valid-return-01", "Alternate tools"),
    ("linked-rollin-02", "Tool histories"),
    ("followup-protected-01", "Follow-ups"),
    ("visit-residual-01", "Visit readout"),
    ("integrated-assistance-01", "Consolidation"),
)


def build():
    rows = []
    for study, label in STUDIES:
        root = Path("research/results") / study
        if (
            study == "integrated-assistance-01"
            and not Path(
                "docs/measurements/integrated-assistance-reading.json"
            ).exists()
        ):
            continue
        a, b = [unpack(root / (arm + ".json.gz")) for arm in ("candidate", "control")]
        items = []
        for left, right in zip(a, b, strict=True):
            assert left["id"] == right["id"]
            if left["family"] == "open":
                continue
            for l, r in zip(left["requests"], right["requests"], strict=True):
                sa, sb = l["score"], r["score"]
                items.append(
                    {
                        "family": left["family"],
                        "candidate_completed": sa["completed"],
                        "control_completed": sb["completed"],
                        "candidate_grounded": sa["success"],
                        "control_grounded": sb["success"],
                    }
                )
        sub = {
            r["completion"]: r for r in completed_pairs(items) if r["family"] == "all"
        }

        deltas = {group: q["gains"] - q["losses"] for group, q in sub.items()}

        whole = sum(
            int(x["candidate_grounded"]) - int(x["control_grounded"]) for x in items
        )
        common = deltas.get("Both completed", 0)
        one = deltas.get("Loop only", 0) + deltas.get("Base only", 0)
        assert deltas.get("Neither completed", 0) == 0 and whole == one + common
        both = sub.get("Both completed", {})
        rows.append(
            {
                "study": study,
                "label": label,
                "requests": len(items),
                "both_finished": both.get("requests", 0),
                "both_loop_correct": both.get("loop_correct", 0),
                "both_base_correct": both.get("base_correct", 0),
                "net_common": common,
                "net_one_sided": one,
                "net_all": whole,
                "archive": digest(root / "checksums.json"),
            }
        )
    target = Path("paper")
    write_csv(target / "assistance-completion-history.csv", rows)
    with plt.rc_context(
        {
            "svg.hashsalt": "loopqwen-completion-history",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        fig, ax = plt.subplots(figsize=(6.8, 2.25), layout="constrained")
        x = np.arange(len(rows))
        for sign, field, label, color in (
            (-1, "net_common", "Both delivered a visible reply", "#146958"),
            (1, "net_one_sided", "Only one delivered a reply", "#a76932"),
        ):
            bars = ax.bar(
                x + sign * 0.19,
                [100 * r[field] / r["requests"] for r in rows],
                width=0.35,
                label=label,
                color=color,
            )
            ax.bar_label(
                bars, labels=[str(r[field]) for r in rows], padding=2, fontsize=7
            )
        ax.axhline(0, lw=0.65, color="#333333")
        ax.set(
            xticks=x,
            xticklabels=[r["label"] for r in rows],
            ylabel="Net gains / objective requests (%)",
        )
        ax.legend(frameon=False, fontsize=7)
        fig.savefig(
            target / "assistance-completion-history.svg", metadata={"Date": None}
        )
        plt.close(fig)
    return rows


if __name__ == "__main__":
    for r in build():
        print(r["study"], r["requests"], r["net_common"], r["net_one_sided"])
