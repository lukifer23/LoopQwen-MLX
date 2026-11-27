"""Trace the path from an available real tool to a delivered grounded answer."""

from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import write_documents
from loopqwen.research.integrated_assistance import providers
from loopqwen.research.native_function_study import executions
from scripts.assistance_deliberation import write_csv

ROOT = Path("research/results/integrated-assistance-01")


def plot(reports, outcome):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with plt.rc_context(
        {
            "font.size": 9,
            "svg.hashsalt": "integrated-tool-path",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        fig, (acts, quality) = plt.subplots(
            1,
            2,
            figsize=(8.4, 2.65),
            layout="constrained",
            gridspec_kw={"width_ratios": [1.35, 1]},
        )
        labels = [
            "Callback executed",
            "Returned data",
            "Visible answer",
            "Grounded answer",
        ]
        metrics = [
            "observed_call",
            "nonerror_observation",
            "visible_answer",
            "grounded_answer",
        ]
        for j, (arm, name, color) in enumerate(
            (
                ("candidate", "Shared loop", "#176859"),
                ("control", "Untouched Base", "#765487"),
            )
        ):
            c = reports[arm]["all"]
            bars = acts.barh(
                np.arange(4) + (j - 0.5) * 0.36,
                [c[k] for k in metrics],
                height=0.34,
                label=name,
                color=color,
            )
            acts.bar_label(bars, fontsize=7, padding=2)
            bars = quality.barh(
                np.arange(2) + (j - 0.5) * 0.36,
                [outcome[arm][k]["correct"] for k in ("tools", "plain")],
                height=0.34,
                color=color,
            )
            quality.bar_label(bars, fontsize=7, padding=2)
        acts.set(
            yticks=np.arange(4),
            yticklabels=labels,
            xlabel="Requests out of 106 tool-enabled",
            xlim=(0, 113),
        )
        quality.set(
            yticks=np.arange(2),
            yticklabels=["With tools (106)", "Without tools (102)"],
            xlabel="Correct objective answers",
            xlim=(0, 57),
        )
        for ax in (acts, quality):
            ax.invert_yaxis()
        acts.legend(loc="lower right", frameon=False, fontsize=7)
        fig.savefig("paper/assistance-tool-path.svg", metadata={"Date": None})
        plt.close(fig)


def build():
    spec = read(ROOT / "registration.json")
    panel = read(spec["material"])
    rows = panel["rows"]
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        write_documents(panel, folder)
        catalogues = [{t.name for t in providers(w, folder)} for w in rows]
    data = []
    reports = {}
    for side in ("candidate", "control"):
        runs = unpack(ROOT / (side + ".json.gz"))
        counts = defaultdict(Counter)
        for w, names, r in zip(rows, catalogues, runs, strict=True):
            assert r["id"] == w["id"]
            (q,) = r["requests"]
            tr = q["trajectory"]
            events = executions(tr)
            assert all(e["call"]["name"] in names for e in events)
            if not names:
                assert not events
                continue
            ok = [e for e in events if "error" not in e["observation"]]
            errors = [j for j, e in enumerate(events) if "error" in e["observation"]]
            permits = [
                j for j, e in enumerate(events) if "error" not in e["observation"]
            ]
            canonical = [fingerprint(e["call"]) for e in events]
            outcome = q["score"]["success"]
            assert not outcome or ok
            common = {
                "requests": 1,
                "observed_call": int(bool(events)),
                "nonerror_observation": int(bool(ok)),
                "visible_answer": int(tr["status"] == "answered"),
                "grounded_answer": int(outcome),
                "obtained_observation_but_missed": int(bool(ok) and not outcome),
                "invalid_call_terminal": int(tr["status"] == "invalid_call"),
                "callback_error": int(bool(errors)),
                "successful_callback_after_error": int(
                    bool(errors and permits and max(permits) > min(errors))
                ),
                "grounded_after_error": int(bool(errors) and outcome),
                "repeated_identical_call": int(len(set(canonical)) < len(canonical)),
                "executed_calls": len(events),
                "nonerror_calls": len(ok),
            }
            for group in (w["family"], "all"):
                counts[group].update(common)
            data.append(
                {
                    "id": w["id"],
                    "family": w["family"],
                    "arm": side,
                    "catalogue": ",".join(sorted(names)),
                    **common,
                    "status": tr["status"],
                    "total_s": q["wall_s"],
                    "first_s": q["first_s"],
                }
            )
        reports[side] = {g: dict(c) for g, c in counts.items()}
    overall = read(ROOT / "decision.json")["arms"]
    outcome = {}
    for arm in ("candidate", "control"):
        tool = reports[arm]["all"]
        total = overall[arm]["scored"]
        elapsed = sum(r["total_s"] for r in data if r["arm"] == arm)
        outcome[arm] = {
            "tools": {
                "requests": tool["requests"],
                "correct": tool["grounded_answer"],
                "wall_s": elapsed,
            },
            "plain": {
                "requests": total["requests"] - tool["requests"],
                "correct": total["grounded"] - tool["grounded_answer"],
                "wall_s": total["wall_s"] - elapsed,
            },
        }
    plot(reports, outcome)
    path = Path("paper/assistance-tool-path.csv")
    write_csv(path, data)
    result = {
        "archive": digest(ROOT / "checksums.json"),
        "rows": digest(path),
        "totals": reports,
        "objective_by_catalogue": outcome,
        "definition": "Reconstruct the tools actually supplied on each original request. A nonerror observation is an executed callback returning data without an error field. It can still be empty, irrelevant, truncated or insufficient. Grounding uses the original study's stricter per-family rule.",
    }
    dest = Path("docs/measurements/integrated-assistance-tool-path.json")
    if dest.exists():
        assert read(dest) == result
    else:
        save(dest, result)
    print({k: v["all"] for k, v in reports.items()})


if __name__ == "__main__":
    build()
