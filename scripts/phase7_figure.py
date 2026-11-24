"""Plot frozen-cohort exposure and actual depth extrapolation from archived predictions."""

import argparse
import gzip
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(root, out):
    def data(path):
        return json.loads(gzip.decompress(path.read_bytes()))["records"]

    parent = Path("docs/measurements/phase6/randomized")
    step = json.loads((root / "decision.json").read_bytes())["step"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for path, deeper, label, color in [
        (parent / "orders-480.json.gz", parent / "deep.json.gz", "R480", "#d47e1d"),
        (root / "endpoint.json.gz", root / "deep.json.gz", f"R{step}", "#00787b"),
    ]:
        rows = data(path)
        ys = []
        for j in range(3):
            s = [r for r in rows if r["condition"].startswith("random")]
            ys.append(
                100
                * sum(
                    r["gold"][j] == max(range(8), key=r["probabilities"][j].__getitem__)
                    for r in s
                )
                / len(s)
            )
        axes[0].plot(range(1, 4), ys, color=color, label=label, marker="o")
        rows = [r for r in data(deeper) if r["condition"] == "random0"]
        ys = [
            100
            * sum(
                r["gold"][j] == max(range(8), key=r["probabilities"][j].__getitem__)
                for r in rows
            )
            / len(rows)
            for j in range(6)
        ]
        axes[1].plot(range(1, 7), ys, color=color, label=label, marker="o")
    axes[0].set(
        title="Trained depths · eight fixed shuffles", xticks=[1, 2, 3], ylim=(90, 101)
    )
    axes[1].set(
        title="Depth extrapolation · fixed shuffle 0", xticks=range(1, 7), ylim=(0, 103)
    )
    for a in axes:
        a.set(xlabel="Shared block applications", ylabel="Correct state (%)")
        a.grid(axis="y", alpha=0.2)
        a.legend(frameon=False)
    fig.savefig(out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, default=Path("docs/measurements/phase7"))
    p.add_argument(
        "--output", type=Path, default=Path("docs/measurements/phase7-exposure.png")
    )
    a = p.parse_args()
    plot(a.archive, a.output)
