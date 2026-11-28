"""Build the direct LoopQwen versus untouched Base paper from verified native records."""

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pypdf import PdfReader
import typst

from loopqwen.io import digest, read, save
from scripts.middle_depth_budget import ROOT

P = Path("paper")
SNAPSHOT = P / "paper-draft-data.json"
ARMS = ("base", "loop")
NAMES = {"base": "Untouched Base", "loop": "LoopQwen"}
FAMILIES = ("reasoning", "tools", "chat", "documents")
LABELS = ("Reasoning", "Executed tools", "Chat", "Full documents")


def snapshot():
    hashes = {}

    def use(path):
        path = Path(path)
        hashes[str(path)] = digest(path)
        return read(path)

    data = {
        "authors": ["lukifer23", "GPT 6.2 Astra"],
        "title": "LoopQwen versus Qwen Base",
        "arms": list(ARMS),
    }
    for label, file in (
        ("hardware", "hardware.json"),
        ("local", "local-workflow-screen.json"),
        ("replay", "standalone/pair.json"),
        ("direct_audit", "direct/parent-audit.json"),
        ("restore", "standalone/restore.json"),
        ("source_audit", "data-audit.json"),
    ):
        data[label] = use(ROOT / file)
    data["specialist"] = (
        use(ROOT / "specialist-result.json")
        if (ROOT / "specialist-result.json").is_file()
        else None
    )
    data["fit"] = next(
        r for r in use(ROOT / "fit-work-0576.json") if r["arm"] == "loop"
    )
    trained = use(ROOT / "terminal-fit-integrity.json")
    data["weights"] = trained["models"]["loop"]
    data["donor"] = trained["matched_identity"]["base_text"]
    data["train"] = use("research/sources/middle-depth/inventory.json")["quotas"]
    data["format"] = use(ROOT / "training-format-audit.json")["families"]
    actions = use(ROOT / "direct/training-first-actions.json")
    if actions["data"] != digest(
        Path("research/sources/middle-depth/training.json.gz")
    ):
        raise ValueError("Initial tool actions come from a different fitting mixture")
    data["training_actions"] = {k: v for k, v in actions.items() if k != "rows"}
    data["recall"] = [
        r
        for r in use(ROOT / "parent-evidence-recall.json")["summary"]
        if r["arm"] in ARMS
    ]
    data["novels"] = []
    books = use(ROOT / "document-review/parent/scored.json")
    caps = {}
    for cap in (128, 512):
        comparison = use(ROOT / f"direct/parent-{cap}.json")
        assert comparison["models"] == ["loop", "base"]
        x = comparison["product"]
        for row in x["rows"]:
            for _, _, relative, sha, _ in (row["left"], row["right"]):
                path = ROOT / relative
                if digest(path) != sha:
                    raise ValueError(f"Changed measured trajectory {path}")
                hashes[str(path)] = sha
        caps[str(cap)] = {
            k: x[k]
            for k in (
                "macro_delta",
                "paired_interval",
                "deadline_macro_delta",
                "families",
                "quality_by_deadline",
            )
        }
        caps[str(cap)]["thresholds"] = comparison["development_thresholds"]
        caps[str(cap)]["sources"] = []
        for family in FAMILIES:
            for source in x["families"][family]["sources"]:
                subset = [
                    r
                    for r in x["rows"]
                    if (r["family"], r["source"]) == (family, source)
                ]
                caps[str(cap)]["sources"].append(
                    dict(
                        family=family,
                        source=source,
                        n=len(subset),
                        base=sum(r["right"][0] for r in subset),
                        loop=sum(r["left"][0] for r in subset),
                        gained=[r["world"] for r in subset if r["delta"] == 1],
                        lost=[r["world"] for r in subset if r["delta"] == -1],
                    )
                )
        for r in books:
            if (
                r["arm"] in ARMS
                and r["cap"] == cap
                and r["kind"] == "first"
                and r["world"] in ("original", "edited")
            ):
                data["novels"].append(
                    {
                        k: r[k]
                        for k in (
                            "arm",
                            "cap",
                            "source",
                            "world",
                            "status",
                            "input_tokens",
                            "grade",
                            "seconds_cold",
                            "seconds_warm",
                            "first_visible_cold",
                            "peak",
                            "scratch_bytes",
                        )
                    }
                )
    data["caps"] = caps
    short = use(ROOT / "review/terminal/scored.json")
    data["serving"] = {}
    for arm in ARMS:
        part = [
            r
            for r in short
            if r["arm"] == arm
            and r["cap"] == 512
            and r["world"] in ("original", "changed", "irrelevant")
        ]
        assert len(part) == 27
        data["serving"][arm] = dict(
            seconds=sum(r["seconds"] for r in part),
            tokens=sum(r["token_count"] for r in part),
            peak=max(r["peak"] for r in part),
            visible=sum(r["first_visible"] is not None for r in part),
            correct=sum(r["grade"]["correct"] for r in part),
            by_deadline=sum(r["grade"]["correct"] and r["seconds"] <= 8 for r in part),
            mean_time_to_correct=fmean(
                min(8, r["seconds"]) if r["grade"]["correct"] else 8 for r in part
            ),
        )
    past = use("research/results/native-induction-anchored-longbase-01/decision.json")
    data["historical_induction"] = past["scores"]
    use(ROOT / "standalone/base-files.json")
    use(ROOT / "standalone/loop-files.json")
    closure = use(ROOT / "ordinary-full-fit-closure.json")
    cutoff = read(ROOT / "charges" / closure["charge"])["dispatch"]
    if data["specialist"]:
        cutoff = max(
            cutoff,
            max(
                read(p)["dispatch"]
                for p in (ROOT / "charges").glob("*.json")
                if read(p)["job"].startswith("specialist-transfer/")
            ),
        )
    public = ROOT / "standard-five"
    data["standard"] = None
    if (public / "result.json").is_file():
        data["standard"] = use(public / "result.json")
        use(public / "source-audit.json")
        use(public / "tokens.json")
        use(public / "enrollment.json")
        for arm in ARMS:
            receipt = use(public / (arm + "-process.json"))
            grades_path = public / (arm + "-grades.json")
            grades = use(grades_path)
            if digest(grades_path) != receipt["grades"] or len(grades) != 250:
                raise ValueError("Standard benchmark score receipt changed")
            for task in data["standard"]["benchmarks"]:
                path = public / arm / (task + ".json")
                raw = use(path)
                subset = [r for r in grades if r["key"].split("/")[0] == task]
                if (
                    len(raw) != 50
                    or len(subset) != 50
                    or any(r["receipt"] != digest(path) for r in subset)
                ):
                    raise ValueError("Incomplete standard benchmark likelihoods")
            cutoff = max(cutoff, receipt["exited"])
    if (public / "depth-result.json").is_file():
        depth = use(public / "depth-result.json")
        receipt = use(public / "loop-k1-process.json")
        if receipt["grades"] != digest(public / "loop-k1-grades.json"):
            raise ValueError("One-pass standard grades changed")
        grades = use(public / "loop-k1-grades.json")
        for task in depth["benchmarks"]:
            path = public / "loop-k1" / (task + ".json")
            subset = [r for r in grades if r["key"].split("/")[0] == task]
            if (
                len(use(path)) != 50
                or len(subset) != 50
                or any(r["receipt"] != digest(path) for r in subset)
            ):
                raise ValueError("Missing same-weight standard per-item likelihoods")
        data["standard_depth"] = depth
        cutoff = max(cutoff, receipt["exited"])
    selected = ROOT / "direct-decision.json"
    if selected.is_file():
        selection = use(selected)
        child = use(ROOT / "long-fit-loop.json")
        resume = use(ROOT / "full-fit-resume-replay.json")
        quantities = use(ROOT / "direct/quantity-review.json")
        initial_calls = use(ROOT / "direct/first-calculations.json")
        use(ROOT / "direct/callback-replay.json")
        use(ROOT / "document-review/direct-child/grades.json")
        use(ROOT / "review/direct-full-fit/grades.json")
        for stage in ("document-review/direct-child", "review/direct-full-fit"):
            for row in use(ROOT / stage / "scored.json"):
                path = ROOT / row["file"]
                if digest(path) != row["sha"]:
                    raise ValueError(f"Changed full-document fit trajectory {path}")
                hashes[str(path)] = row["sha"]
        data["full_fit"] = {
            "decision": selection,
            "weights": child["weights"],
            "steps": len(child["log"]),
            "first_losses": child["log"][0]["loss"],
            "last_losses": child["log"][-1]["loss"],
            "positions": child["positions"],
            "features": use(ROOT / "long-features-loop.json"),
            "resume_passed": resume["passed"],
            "quantity_rows": len(quantities["rows"]),
            "first_calls": len(initial_calls["rows"]),
        }
        cutoff = max(
            cutoff,
            max(
                r["dispatch"] + r["seconds"]
                for p in (ROOT / "charges").glob("*.json")
                if (r := read(p))["job"].endswith("/direct-child")
                and r["status"] == "finished"
            ),
        )
    probe = ROOT / "direct/short-depth-probe.json"
    if probe.is_file():
        data["depth"] = use(probe)
        use(ROOT / "review/depth/grades.json")
        for row in use(ROOT / "review/depth/scored.json"):
            path = ROOT / row["file"]
            if digest(path) != row["sha"]:
                raise ValueError(f"Changed within-checkpoint answer {path}")
            hashes[str(path)] = row["sha"]
        cutoff = max(
            cutoff,
            max(
                r["dispatch"] + r["seconds"]
                for p in (ROOT / "charges").glob("*.json")
                if "/short/depth/" in (r := read(p))["job"]
                and r["status"] == "finished"
            ),
        )
    charges = defaultdict(float)
    installed = ROOT / "standalone/selected-benchmark"
    if (installed / "audit.json").is_file():
        record = use(installed / "audit.json")
        scores = use(installed / "result.json")
        assert record["result"] == digest(installed / "result.json")
        for label, run in record["runs"].items():
            cap, arm = label.split("/")
            trace_file = installed / cap / (arm + "-replay.json.gz")
            if (
                digest(trace_file) != run["trace_sha256"]
                or run["trace_sha256"] != scores["runs"][label]["trace"]
            ):
                raise ValueError("Changed installed trace")
            hashes[str(trace_file)] = run["trace_sha256"]
            cutoff = max(cutoff, run["exited"])
        data["installed_selected"] = record
    long_probe = ROOT / "direct/book-recall-depth.json"
    if long_probe.is_file():
        data["book_depth"] = use(long_probe)
        use(ROOT / "direct/short-depth-uncertainty.json")
        use(ROOT / "document-review/recall-depth/fact-grades.json")
        use(ROOT / "document-review/recall-depth/grades.json")
        for pair in data["book_depth"]["rows"].values():
            for r in pair.values():
                path = ROOT / r["trace"]
                if digest(path) != r["sha"]:
                    raise ValueError("Changed complete-book pass intervention")
                hashes[str(path)] = r["sha"]
        cutoff = max(
            cutoff,
            max(
                r["dispatch"] + r["seconds"]
                for p in (ROOT / "charges").glob("*.json")
                if (r := read(p))["job"].endswith("/recall-depth")
                and r["status"] == "finished"
            ),
        )
    book_panel = ROOT / "direct/primary-books.json"
    if book_panel.is_file():
        product = use(book_panel)
        review_dir = ROOT / "document-review/direct-primary-books"
        grades = review_dir / "grades.json"
        scored = review_dir / "scored.json"
        if digest(scored) != product["source"] or digest(grades) != product["grades"]:
            raise ValueError("Complete-book grades changed")
        use(grades)
        rows = use(scored)
        if len(rows) != product["cases"] or len(rows) != 48:
            raise ValueError("Incomplete primary book panel")
        for r in rows:
            path = ROOT / r["file"]
            if digest(path) != r["sha"]:
                raise ValueError(f"Changed full-book conversation {path}")
            hashes[str(path)] = r["sha"]
        data["primary_books"] = {
            k: v["groups"] for k, v in product["contracts"].items()
        }
        cutoff = max(
            cutoff,
            max(
                r["dispatch"] + r["seconds"]
                for p in (ROOT / "charges").glob("*.json")
                if (r := read(p))["stage"] == "long"
                and r["job"].endswith("/parent")
                and r["status"] == "finished"
            ),
        )
    for p in sorted((ROOT / "charges").glob("*.json")):
        r = read(p)
        if r["dispatch"] <= cutoff and r["status"] != "running":
            use(p)
            charges[r["stage"]] += r["seconds"]
    data["charges"] = charges
    data["input_files"] = hashes
    save(SNAPSHOT, data)


def figures(d, folder):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#253c47",
            "svg.fonttype": "none",
            "svg.hashsalt": "loopqwen-direct-pair",
        }
    )
    colors = {"base": "#536b86", "loop": "#117869"}
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 2.7), sharey=True, layout="constrained")
    for ax, cap in zip(axs, ("128", "512"), strict=True):
        for i, (arm, side) in enumerate((("base", "right"), ("loop", "left"))):
            vals = [100 * d["caps"][cap]["families"][f][side] for f in FAMILIES]
            ax.barh(
                [3 - j + (i - 0.5) * 0.34 for j in range(4)],
                vals,
                height=0.32,
                color=colors[arm],
                label=NAMES[arm],
            )
        ax.set_yticks([3, 2, 1, 0], LABELS)
        ax.set_xlim(0, 51)
        ax.set_title(cap + " output tokens / turn", fontsize=10)
        ax.set_xlabel("Supported answers (%)")
        ax.grid(axis="x", alpha=0.17)
        ax.set_axisbelow(True)
    axs[1].legend(frameon=False, fontsize=8)
    fig.savefig(folder / "quality.svg", metadata={"Date": None})
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 2.7), layout="constrained")
    result = d["caps"]["512"]["quality_by_deadline"]
    for ax, long in zip(axs, (False, True), strict=True):
        families = ["documents"] if long else ["chat", "reasoning", "tools"]
        xs = sorted(map(int, result[families[0]]))
        for arm, side in (("base", "right"), ("loop", "left")):
            ys = [100 * fmean(result[f][str(x)][side] for f in families) for x in xs]
            ax.step(
                xs, ys, where="post", label=NAMES[arm], color=colors[arm], linewidth=2
            )
            ax.scatter(xs, ys, color=colors[arm], s=14)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [str(x) for x in xs], fontsize=8)
        ax.set_ylim(-2, 31)
        ax.set_ylabel("Correct by deadline (%)")
        ax.set_xlabel("Deadline (seconds)")
        ax.set_title("Complete books, cold" if long else "Short requests", fontsize=10)
        ax.grid(alpha=0.17)
    axs[0].legend(frameon=False, fontsize=8)
    fig.savefig(folder / "deadline.svg", metadata={"Date": None})
    plt.close(fig)


def render():
    d = read(SNAPSHOT)
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        figures(d, folder)
        (folder / "data.json").write_text(json.dumps(d))
        (folder / "paper.typ").write_bytes((P / "paper-draft.typ").read_bytes())
        pdf = typst.compile(str(folder / "paper.typ"), root=str(folder), timestamp=0)
        path = folder / "proof.pdf"
        path.write_bytes(pdf)
        pages = [p.extract_text() for p in PdfReader(path).pages]
        if not (
            6 <= len(pages) <= 14
            and all(len(p) > 560 for p in pages)
            and all(a in pages[0] for a in d["authors"])
        ):
            raise ValueError(
                f"Inspect PDF pagination and byline: {[len(p) for p in pages]}"
            )
        # Publish the plotted data and the exact artwork embedded in this draft.
        (P / "paper-draft-quality.svg").write_bytes(
            (folder / "quality.svg").read_bytes()
        )
        (P / "paper-draft-deadlines.svg").write_bytes(
            (folder / "deadline.svg").read_bytes()
        )
        with (P / "paper-draft-sources.csv").open("w", newline="") as out:
            writer = csv.writer(out, lineterminator="\n")
            writer.writerow(
                (
                    "output_cap",
                    "family",
                    "source",
                    "openings",
                    "base_correct",
                    "loop_correct",
                    "gained_worlds",
                    "lost_worlds",
                )
            )
            for cap, values in d["caps"].items():
                for row in values["sources"]:
                    writer.writerow(
                        (
                            cap,
                            row["family"],
                            row["source"],
                            row["n"],
                            row["base"],
                            row["loop"],
                            "|".join(row["gained"]),
                            "|".join(row["lost"]),
                        )
                    )
        (P / "paper-draft.pdf").write_bytes(pdf)
        print(
            len(pages),
            "pages",
            len(d["input_files"]),
            "evidence digests",
            digest(P / "paper-draft.pdf"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    if parser.parse_args().refresh:
        snapshot()
    render()
