"""Reconstruct paired evidence, figures, and the current complete manuscript."""

import csv
import gzip
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from assemble_field_paper import assemble

from loopqwen.io import digest, read
from loopqwen.research.native_function_study import executions
from loopqwen.research.role_binding import material

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
NAME = "role-binding"
TRIAL = ROOT / "research/results/role-binding-01"
SIDES = ("candidate", "control")


def write_csv(suffix, rows):
    with (PAPER / (NAME + "-" + suffix + ".csv")).open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=list(dict.fromkeys(k for r in rows for k in r))
        )
        w.writeheader()
        w.writerows(rows)


def plot(metrics):
    matplotlib.rcParams.update(
        {"svg.hashsalt": NAME, "font.family": "DejaVu Sans", "font.size": 9}
    )
    fig, ax = plt.subplots(figsize=(7.5, 3.0), layout="constrained")
    for j, (s, label, color, offset) in enumerate(
        [("Loop", "Loop", "#146456", -0.17), ("Base", "Base", "#956035", 0.17)]
    ):
        for i, r in enumerate(metrics):
            pct = 100 * r[s] / r["N"]
            ax.barh(i + offset, pct, 0.30, color=color, label=label if i == 0 else None)
            ax.text(pct + 1.3, i + offset, f"{r[s]}/{r['N']}", va="center", fontsize=8)
    ax.set_yticks(range(len(metrics)), [r["Endpoint"] for r in metrics])
    ax.invert_yaxis()
    ax.set_xlim(0, 117)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share solved (%)")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(PAPER / (NAME + ".svg"), metadata={"Date": None})
    plt.close(fig)


def cost_figure(rows):
    fig, ax = plt.subplots(figsize=(6.1, 3.25), layout="constrained")
    styles = (
        (True, False, "Loop only", "#167264", "o"),
        (False, True, "Base only", "#aa5b32", "^"),
        (True, True, "Both correct", "#405b94", "s"),
        (False, False, "Both failed", "#9b9b9b", "x"),
    )
    for a, b, label, color, marker in styles:
        selected = [
            r for r in rows if (r["candidate_success"], r["control_success"]) == (a, b)
        ]
        ax.scatter(
            [r["control_wall_s"] for r in selected],
            [r["candidate_wall_s"] for r in selected],
            s=16,
            alpha=0.8,
            color=color,
            marker=marker,
            label=f"{label} ({len(selected)})",
        )
    limits = [
        min(r[k] for r in rows for k in ("control_wall_s", "candidate_wall_s")) * 0.75,
        max(r[k] for r in rows for k in ("control_wall_s", "candidate_wall_s")) * 1.23,
    ]
    ax.plot(limits, limits, color="#727272", lw=0.8, ls="--")
    ax.set(
        xscale="log",
        yscale="log",
        xlim=limits,
        ylim=limits,
        xlabel="Ordinary Base wall time (seconds)",
        ylabel="Recurrent wall time (seconds)",
    )
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(PAPER / (NAME + "-times.svg"), metadata={"Date": None})
    plt.close(fig)


def main():
    decision = read(TRIAL / "decision.json")
    spec = read(TRIAL / "registration.json")
    registry = read(ROOT / "research/registry.json")
    assert registry["experiments"][spec["id"]]["archive"] == digest(
        TRIAL / "checksums.json"
    )
    assert all(
        digest(TRIAL / k) == v for k, v in read(TRIAL / "checksums.json").items()
    )
    data = material(spec)
    runs = {
        s: json.loads(gzip.decompress((TRIAL / (s + ".json.gz")).read_bytes()))
        for s in SIDES
    }
    scored = []
    lookup = {s: Counter() for s in SIDES}
    for i, w in enumerate(data["rows"]):
        for j, question in enumerate(w["questions"]):
            row = {
                "id": w["id"],
                "cluster": w["cluster"],
                "family": w["family"],
                "message": j + 1,
                "question": question,
                "gold": w["gold"][j] if w["family"] == "inventory" else w["gold"],
                "document": w.get("document"),
                "option_order": json.dumps(w.get("options"), ensure_ascii=False),
                "expected_lookup": json.dumps(w.get("arguments"), ensure_ascii=False),
            }
            for s in SIDES:
                r = runs[s][i]
                q = r["requests"][j]
                tr = q["trajectory"]
                assert r["id"] == w["id"]
                events = executions(tr)
                ev = {
                    "answer_json": json.dumps(tr["answer"], ensure_ascii=False),
                    "status": tr["status"],
                    **q["score"],
                    "calls_and_returns": json.dumps(events, ensure_ascii=False),
                    "first_s": q["first_s"],
                    "wall_s": q["wall_s"],
                    "peak_bytes": q["peak"],
                    "prompt_tokens": len(tr["turns"][0]["prompt_ids"]),
                    "output_tokens": sum(len(t["token_ids"]) for t in tr["turns"]),
                }
                if w["family"] == "table":
                    first = next(
                        (e for e in events if e["call"]["name"] == "csv_lookup"), None
                    )
                    for key in ("key_column", "key", "value_column"):
                        matches = bool(
                            first
                            and first["call"]["arguments"].get(key)
                            == w["arguments"][key]
                        )
                        ev["first_lookup_correct_" + key] = matches
                        lookup[s][key] += int(matches)
                row.update({s + "_" + k: v for k, v in ev.items()})
            scored.append(row)
    write_csv("requests", scored)
    d = decision["groups"]
    m = []

    def add(label, n, vals):
        m.append({"Endpoint": label, "N": n, "Loop": vals[0], "Base": vals[1]})

    for f, label, n in (
        ("meaning", "Human meaning", 48),
        ("table", "Human table lookup", 32),
        ("inventory", "Package follow-up", 16),
    ):
        add(label + " · requests", n, [d[s][f]["grounded"] for s in SIDES])
        add(
            label + " · complete pairs",
            n // 2,
            [decision["pairs"][f][x] for x in ("loop", "base")],
        )
    write_csv("accuracy", m)
    plot(m)
    cost_figure(scored)
    costs = []
    for label, key, scale in [
        ("Cold stream wall (s)", "wall_s", 1),
        ("Sum of first-token waits (s)", "first_s", 1),
        ("Maximum MLX allocation (GiB)", "peak", 1 / 2**30),
        ("Opening tokens / first seconds", "opening_rate", 1),
        ("Post-first tokens / second", "decode_rate", 1),
        ("Model output tokens", "output_tokens", 1),
    ]:
        vals = [
            (d[s] if key.endswith("_rate") else d[s]["all"])[key] * scale for s in SIDES
        ]
        costs.append(
            {"Measurement": label, "Loop": f"{vals[0]:,.2f}", "Base": f"{vals[1]:,.2f}"}
        )
    write_csv("costs", costs)
    write_csv(
        "argument-roles",
        [
            {
                "Component": k,
                "Loop": lookup["candidate"][k],
                "Base": lookup["control"][k],
                "N": 32,
            }
            for k in ("key_column", "key", "value_column")
        ],
    )
    print(len(scored), "paired requests", lookup)
    assemble(PAPER)


if __name__ == "__main__":
    main()
