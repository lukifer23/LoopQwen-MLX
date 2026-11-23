"""Archive small measurements and the two seed-43 inference adapters after checks pass."""

import gzip
import json
import shutil
from pathlib import Path

from phase2_figures import figure, frontier_plot

from loopqwen.io import checkpoint_location as location
from loopqwen.io import digest, read, save

OUT = Path("outputs/phase2")
FIRST_ARCHIVE = Path("docs/measurements/phase2")
REPRO = (FIRST_ARCHIVE / "sha256.json").exists()
DEST = OUT / "reproduction/measurements/phase2" if REPRO else FIRST_ARCHIVE


def archive(name):
    summary = OUT / name / "comparison.md"
    if not summary.exists():
        raise FileNotFoundError(f"Evaluation group is incomplete: {name}")
    m = read(Path("experiments") / (name + ".json"))
    folder = DEST / name
    folder.mkdir(exist_ok=True, parents=True)
    for arm in m["runs"]:
        weights = location(Path("checkpoints") / arm)
        result = read(OUT / name / (arm + ".json"))
        if result["identity"]["weights"] != digest(weights / "adapters.safetensors"):
            raise ValueError("Checkpoint changed")
        for suffix, data in (("", result), (".training", read(weights / "state.json"))):
            # Full curves, all prompt completions and provenance. Omit bookkeeping of private temp paths.
            (folder / (arm + suffix + ".json.gz")).write_bytes(
                gzip.compress(json.dumps(data, separators=(",", ":")).encode(), mtime=0)
            )
    shutil.copyfile(summary, folder / "comparison.md")
    figure(name, DEST)


def all_results():
    for group in (
        "phase2_direct",
        "phase2_screen",
        "phase2_confirm",
        "phase2_repeat",
        "phase2_anchor",
    ):
        archive(group)
    frontier_plot(read(OUT / "frontier.json"), DEST)
    for key in (
        "frontier",
        "base_dev",
        "transfer",
        "format_shift",
        "gate_sweep",
        "merge_audit",
        "merge_trace",
        "depth_latency",
        "padded",
    ):
        (DEST / (key + ".json.gz")).write_bytes(
            gzip.compress((OUT / (key + ".json")).read_bytes(), mtime=0)
        )
    for key in (
        "repeat",
        "profile",
        "derivatives",
        "compute",
        "validation_memory",
        "padding",
        "integrity",
        "example",
        "anchor_diagnostics",
    ):
        shutil.copyfile(OUT / (key + ".json"), DEST / (key + ".json"))
    shutil.copyfile(
        OUT / "phase2_confirm/paired.json", DEST / "phase2_confirm/paired.json"
    )
    for fusion in ("stock", "fp32"):
        directory = Path("exports") / ("p2_confirm_43_loop_" + fusion)
        for filename in ("verification.json", "provenance.json"):
            shutil.copyfile(directory / filename, DEST / (fusion + "-" + filename))
    for run in ("p2_confirm_43_dense", "p2_confirm_43_loop"):
        src = location(Path("checkpoints") / run)
        dst = (OUT / "reproduction/adapters" if REPRO else Path("adapters")) / run
        dst.mkdir(exist_ok=True, parents=True)
        for filename in ("state.json", "adapters.safetensors"):
            shutil.copyfile(src / filename, dst / filename)
    files = [f for f in DEST.rglob("*") if f.is_file() and f.name != "sha256.json"]
    save(
        DEST / "sha256.json",
        {str(f.relative_to(DEST)): digest(f) for f in sorted(files)},
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        archive(sys.argv[1])
    else:
        all_results()
