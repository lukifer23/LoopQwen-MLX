"""Run the registered studies sequentially on one GPU, selecting only on development data."""

import argparse
import subprocess
import sys
from pathlib import Path

from phase2_configs import write
from phase2_register import register

from loopqwen.io import checkpoint_location as location
from loopqwen.io import read, save

OUT = Path("outputs/phase2")


def execute(path):
    subprocess.run(
        [sys.executable, "-m", "loopqwen", "train", "--config", str(path)], check=True
    )


def assess(name):
    subprocess.run(
        [sys.executable, "-m", "loopqwen", "evaluate", "--experiment", name], check=True
    )


def screening():
    # Same budget for each method. MLP-only isolates span placement across hybrid layer patterns.
    candidates = {
        "direct": {},
        "gradual": {"curriculum": "gradual", "ramp_fraction": 0.25},
        "stabilized": {"mode": "stabilized"},
        "mixed": {"curriculum": "mixed", "depth_weights": [0.2, 0.8, 0]},
        **{
            f"mlp{s}": {
                "start": s,
                "projections": ["up_proj", "down_proj", "gate_proj"],
            }
            for s in (8, 10, 16)
        },
        "rank16": {"rank": 16, "lora_alpha": 32},
        "nearby": {"target": "nearby"},
        "formats": {"mixture": "mixed_answer_and_traces"},
    }
    runs = []
    for label, params in candidates.items():
        path = write("screen_" + label, steps=24, eval_interval=24, **params)
        runs.append(path.stem)
        execute(path)
    control = write(
        "screen_formats_dense",
        steps=24,
        eval_interval=24,
        passes=1,
        mixture="mixed_answer_and_traces",
    )
    execute(control)
    register(
        "phase2_screen",
        runs + [control.stem],
        "Equal-update development screens of curriculum, capacity, MLP-only placement and response-format mixture.",
        [["p2_screen_formats_dense", "p2_screen_formats"]],
    )
    assess("phase2_screen")
    scores = {
        label: read(OUT / "phase2_screen" / (run + ".json"))["scores"]["validation"][
            "answer"
        ]["loss"]["nll"]
        for label, run in zip(candidates, runs)
    }
    best = min(scores, key=scores.get)
    frozen = dict(
        metric="Common answer-only development NLL",
        scores=scores,
        selected=best,
        policy=candidates[best],
        seeds=[43, 53, 67],
        steps=84,
        mixture=candidates[best].get("mixture", "verified_traces"),
        final_id_count=90,
        final_ood_count=90,
        final_trace_count=45,
    )
    path = Path("experiments/phase2_selection.json")
    if path.exists():
        save(OUT / "replayed_selection.json", frozen)
        print(
            "Continuing with previously frozen policy",
            read(path)["selected"],
            flush=True,
        )
    else:
        save(path, frozen)


def confirm():
    design = read("experiments/phase2_selection.json")
    policy = design["policy"]
    runs = []
    pairs = []
    for seed in design["seeds"]:
        common = dict(
            seed=seed,
            steps=design["steps"],
            eval_interval=design["steps"],
            checkpoint_interval=design["steps"],
            **policy,
        )
        dense = common | dict(mode="direct", curriculum="direct", passes=1)
        for label, c in [("dense", dense), ("loop", common)]:
            path = write(f"confirm_{seed}_{label}", **c)
            runs.append(path.stem)
            execute(path)
        pairs.append(runs[-2:])
        s = read(location(Path("checkpoints") / runs[-1]) / "state.json")
        budget = sum(h["seconds"] for h in s["history"])
        path = write(
            f"confirm_{seed}_time",
            **(dense | dict(steps=design["steps"] * 3, seconds_budget=budget)),
        )
        runs.append(path.stem)
        execute(path)
    register(
        "phase2_confirm",
        runs,
        "Frozen three-seed ID and OOD evaluation with data- and time-matched dense controls.",
        pairs,
        final=True,
    )
    assess("phase2_confirm")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("screen", "confirm", "all"))
    stage = p.parse_args().stage
    if stage in ("screen", "all"):
        screening()
    if stage in ("confirm", "all"):
        confirm()
