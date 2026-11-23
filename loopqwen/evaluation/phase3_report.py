from pathlib import Path

from ..io import read


def report():
    root = Path("outputs/phase3")
    main = read(root / "main.json")
    lines = [
        "# Phase 3 development comparison",
        "",
        "Seed 73. Same V2 training order. Greedy traced generation on 60 development items.",
        "",
        "| Arm | Execution | Correct | Verified | Truncated | Final-token NLL | Trace NLL | Prose NLL | Generation seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in main["rows"]:
        lines.append(
            f"| {r['arm']} | {r['mode']} | {r['correct']}/{r['n']} | {r['verified']} | {r['truncated']} | {r['final_nll']:.4f} | {r['trace_nll']:.4f} | {r['retention']:.4f} | {r['seconds']:.1f} |"
        )
    lines += [
        "",
        "Paired changes count fixes and breaks in the model named first relative to its comparator.",
        "",
        "| Pair | Fixes | Breaks | Difference (pp) | Paired item bootstrap 95% (pp) | Exact McNemar p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, d in main["paired"].items():
        lo, hi = d["item_ci95"]
        lines.append(
            f"| {name} | {d['fixes']} | {d['breaks']} | {100 * d['delta']:.1f} | [{100 * lo:.1f}, {100 * hi:.1f}] | {d['mcnemar']:.3f} |"
        )
    path = root / "comparison.md"
    path.write_text("\n".join(lines) + "\n")
    return str(path)
