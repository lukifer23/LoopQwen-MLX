"""Print the primary experimental tables from a verified Phase 6 archive."""

import argparse
import gzip
import json
from pathlib import Path
from statistics import median


def load(p):
    return json.loads(
        gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
    )


def pct(values):
    return " / ".join(f"{v * 100:.1f}" for v in values)


def table(header, rows):
    print("|", " | ".join(header), "|")
    print("|", " | ".join("---" for _ in header), "|")
    for row in rows:
        print("|", " | ".join(map(str, row)), "|")
    print()


def report(root):
    study = load(root / "results.json")
    arms = study["arms"]
    quality = []
    cost = []
    extrap = []
    for name, a in arms.items():
        scores = a["evaluations"][-1]["scores"]
        c = scores["conditions"]["canonical"]
        r = scores["conditions"]["mean_random"]
        quality.append(
            [
                name,
                pct(c["accuracy"]),
                pct(r["accuracy"]),
                f"{100 * r['trajectory'][-1]:.2f}",
                pct(scores["trajectory_ci"]),
                pct(scores["worst_order"]),
            ]
        )
        latency = [r for r in load(root / a["latency"])["results"] if r["depth"] == 3]
        cost.append(
            [
                name,
                a["labels"],
                a["capacity"],
                f"{a['seconds']:.1f}",
                f"{a['maps_s']:.1f}",
                f"{a['labels_s']:.1f}",
                f"{a['peak_gib']:.2f}",
                f"{latency[0]['median_ms']:.1f}",
                f"{latency[1]['maps_s']:.1f}",
            ]
        )
        if a.get("deep"):
            groups = load(root / name / "depth-summary.json")["groups"]
            for order in ["canonical", "random0"]:
                vals = []
                for depth in range(4, 7):
                    g = [
                        s
                        for k, s in groups.items()
                        if k.startswith(f"{order}/{depth}/")
                    ]

                    def fmt(g):
                        return f"{g['correct']}/{g['n']}" if g else "—"

                    vals.append(
                        f"{sum(v['correct'] for v in g)}/384 ({fmt(groups.get(f'{order}/{depth}/new'))}, {fmt(groups.get(f'{order}/{depth}/wrapped'))})"
                    )
                extrap.append([name, order, *vals])
    print("### Accuracy across edge presentations\n")
    table(
        [
            "Training arm",
            "Canonical hops 1 / 2 / 3 (%)",
            "Shuffled hops 1 / 2 / 3 (%)",
            "Shuffled complete path (%)",
            "Map bootstrap 95% interval (%)",
            "Worst order by hop (%)",
        ],
        quality,
    )
    print("### Measured cost\n")
    table(
        [
            "Arm",
            "State labels",
            "Trainable weights",
            "Update seconds",
            "Maps/s",
            "Labels/s",
            "Peak GiB",
            "Batch-one K=3 ms",
            "Batch-eight K=3 maps/s",
        ],
        cost,
    )
    print(
        "### Unseen depth\n\nEach cell gives the total followed by nonwrapping and wrapped counts.\n"
    )
    table(["Arm", "Order", "Hop 4", "Hop 5", "Hop 6"], extrap)
    print("### Ordered and shuffled comparisons\n")
    cond = arms["randomized"]["evaluations"][-1]["scores"]["conditions"]
    table(
        [
            "Presentation",
            "Hit by hop (%)",
            "Full trajectory (%)",
            "Agreement with canonical (%)",
            "Both correct (%)",
            "Correct-state NLL by hop",
            "10-bin ECE by hop",
        ],
        [
            [
                k,
                pct(s["accuracy"]),
                f"{100 * s['trajectory'][-1]:.1f}",
                pct(s["consistent"]),
                pct(s["both_correct"]),
                " / ".join(f"{v:.3f}" for v in s["nll"]),
                " / ".join(f"{v:.3f}" for v in s["ece"]),
            ]
            for k, s in cond.items()
        ],
    )
    for name in ("randomized", "outcome"):
        if name not in arms or "causal" not in arms[name]:
            continue
        result = load(root / arms[name]["causal"])
        print(f"### {name.capitalize()} interventions\n")
        rows = []
        for cond, v in result["interventions"].items():
            for kind in ("cross", "other_start", "same_state"):
                s = v["summary"][f"{kind}/all"]
                rows.append(
                    [
                        cond,
                        kind,
                        *[
                            s[k]
                            for k in (
                                "n",
                                "donor_transition",
                                "natural_donor_transition",
                                "noise_donor_transition",
                                "recipient_persistence",
                                "donor_map_persistence",
                            )
                        ],
                    ]
                )
        table(
            [
                "Donor/recipient order",
                "Donor type",
                "Cases",
                "Patch target",
                "Unpatched target",
                "Noise target",
                "Recipient next state",
                "Donor map successor",
            ],
            rows,
        )
        table(
            ["Counterfactual", "Correct by hop", "Equivariant predictions by hop"],
            [
                [
                    k,
                    " / ".join(str(r["correct"]) for r in g["metrics"]),
                    " / ".join(str(r["prediction_agreement"]) for r in g["metrics"]),
                ]
                for k, g in result["behavior"].items()
            ],
        )
    bench = load(root / arms["randomized"]["benchmark"])
    for name, times in bench["independent_vs_single"].items():
        print(
            "Independent" if name == "True" else "Single unroll",
            round(median(x["seconds"] for x in times) * 1000, 2),
            "ms",
            round(max(x["peak"] for x in times) / 2**30, 2),
            "GiB",
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "archive", type=Path, nargs="?", default=Path("docs/measurements/phase6")
    )
    report(p.parse_args().archive)
