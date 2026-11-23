"""Archive reached milestones, without backbone weights or optimizer generations."""

import argparse
import gzip
import json
import shutil
import statistics
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loopqwen.io import digest, read, save


def json_gzip(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = gzip.compress(json.dumps(value, separators=(",", ":")).encode(), mtime=0)
    if path.exists() and path.read_bytes() != payload:
        raise FileExistsError(
            f"Archive conflict at {path}; use a new publication directory"
        )
    path.write_bytes(payload)


def performance(state):
    h = state["history"]
    secs = sum(r["seconds"] for r in h)
    positions = sum(
        r.get("executed_positions", round(r["inputs"] / (1 - r["padding_fraction"])))
        for r in h
    )
    times = sorted(r["seconds"] for r in h)
    protocols = Counter()
    sources = Counter()
    inputs_by_source = Counter()
    for r in h:
        protocols.update(r.get("examples_by_source", {}))
        sources.update(r["scored_by_source"])
        inputs_by_source.update(r.get("inputs_by_source", {}))
    return dict(
        step=state["step"],
        target_tokens=state["loss_tokens"],
        input_tokens=state["input_tokens"],
        order_hash=state["order_hash"],
        execution_events=state.get("execution_events", []),
        checkpoint_segments={
            str(setting): dict(
                updates=len(part),
                targets=sum(r["labels"] for r in part),
                seconds=sum(r["seconds"] for r in part),
                peak=max(r["peak_bytes"] for r in part),
            )
            for setting in (True, False)
            if (
                part := [
                    r
                    for r in h
                    if r.get("checkpointing", state["config"]["gradient_checkpointing"])
                    == setting
                ]
            )
        },
        unique=len(state["visits"]),
        repeats=sum(r["repeats"] for r in h),
        examples_by_source=dict(protocols),
        targets_by_source=dict(sources),
        inputs_by_source=dict(inputs_by_source),
        eos_tokens=sum(state["visits"].values()),
        executed_positions=positions,
        update_seconds=secs,
        supervised_tps=state["loss_tokens"] / secs,
        input_tps=state["input_tokens"] / secs,
        positions_s=positions / secs,
        updates_s=len(h) / secs,
        seconds_per_million=1e6 * secs / state["loss_tokens"],
        median_s=statistics.median(times),
        p10_s=times[int(0.1 * (len(times) - 1))],
        p90_s=times[int(0.9 * (len(times) - 1))],
        padding=1 - state["input_tokens"] / positions,
        peak_bytes=max(r["peak_bytes"] for r in h),
        rss_peak=max(r["rss"] for r in h),
        min_available=min(r["available"] for r in h),
        swap_start=h[0]["swap"],
        swap_end=h[-1]["swap"],
        swap_peak=max(r["swap"] for r in h),
        alpha=h[-1]["alpha"],
        gate_delta=h[-1].get("gate_delta"),
        gate_gradient=h[-1].get("gate_gradient"),
        validation=[
            dict(
                step=r["step"],
                targets=sum(v["labels"] for v in h[: r["step"]]),
                **r["validation"],
            )
            for r in h
            if "validation" in r
        ],
    )


def publish(destination):
    out = Path("outputs/phase3")
    selection = read(out / "selection.json")
    main = read(out / "main.json")
    dest = Path(destination)
    adapter_dest = Path("adapters/phase3")
    if (dest / "checksums.json").is_file():
        dest = Path("outputs/phase3/reproduction") / digest(out / "main.json")[:16]
        adapter_dest = dest / "adapters"
    budgets = read(out / "exposure.json")
    dest.mkdir(exist_ok=True, parents=True)
    metrics = {}
    index = {}
    for cfg in sorted((out / "configs").glob("*.yaml")):
        root = Path("checkpoints") / cfg.stem
        if not (root / "latest.json").is_file():
            continue
        pointers = [
            (
                cfg.stem,
                "targets-100000.json"
                if (root / "targets-100000.json").exists()
                else "latest.json",
            )
        ]
        if (root / "targets-250000.json").exists():
            pointers.append((cfg.stem + "_250k", "targets-250000.json"))
        for label, pointer in pointers:
            gen = root / read(root / pointer)["directory"]
            state = read(gen / "state.json")
            metrics[label] = performance(state)
            index[label] = dict(
                git=state["git"],
                config=state["config"],
                dataset=state["sampling"],
                source=state.get("training_identity"),
                status=state.get("status"),
                generation=gen.name,
                milestone_reached=pointer.removesuffix(".json")
                if pointer.startswith("targets-")
                else None,
                metadata=digest(gen / "state.json"),
                adapters=digest(gen / "adapters.safetensors"),
            )
            json_gzip(dest / "curves" / (label + ".json.gz"), state["history"])
    save(dest / "training.json", metrics)
    save(dest / "registry.json", index)
    for name in (
        "screen.json",
        "main.json",
        "promotion.json",
        "selection.json",
        "learn1-lr.json",
        "learn2-lr.json",
        "fresh-development.json",
        "native-audit.json",
        "v3-manifest.json",
        "retention-manifest.json",
        "export-audit.json",
        "exposure.json",
        "answer-likelihood.json",
        "learning-curves.json",
        "regression.json",
        "compaction.json",
        "projection.json",
        "fork-audit.json",
        "gate-controls.json",
    ):
        if (out / name).is_file():
            shutil.copyfile(out / name, dest / name)
    shutil.copyfile(out / "schedule.jsonl.gz", dest / "schedule.jsonl.gz")
    if (out / "system.jsonl").exists():
        shutil.copyfile(out / "system.jsonl", dest / "system.jsonl")
    for name in ("frontier.json", "difficulty.json"):
        if (out / name).exists():
            json_gzip(dest / (name + ".gz"), read(out / name))
    for file in out.glob("batch-*.json"):
        shutil.copyfile(file, dest / file.name)
    for file in (out / "eval").rglob("*.json"):
        json_gzip(
            dest
            / "evaluations"
            / file.relative_to(out / "eval").with_suffix(".json.gz"),
            read(file),
        )
    publications = [(arm, 100000) for arm in ("dense", "direct", selection["selected"])]
    if (out / "promotion.json").is_file():
        publications += [(arm, 250000) for arm in ("dense", selection["selected"])]
    for arm, budget in publications:
        exposure = budgets[str(budget)]
        name = "p3_" + arm
        root = Path("checkpoints") / name
        gen = root / read(root / f"targets-{budget}.json")["directory"]
        published = adapter_dest / (name if budget == 100000 else name + "_250k")
        published.mkdir(exist_ok=True, parents=True)
        state = read(gen / "state.json")
        if (state["order_hash"], state["loss_tokens"], state["input_tokens"]) != (
            exposure["order"],
            exposure["targets"],
            exposure["inputs"],
        ):
            raise ValueError("Measured cohort differs from the registered schedule")
        for filename in ("adapters.safetensors",):
            if (published / filename).exists() and digest(
                published / filename
            ) != digest(gen / filename):
                raise FileExistsError(published / filename)
            shutil.copyfile(gen / filename, published / filename)
        save(
            published / "state.json",
            {
                k: state[k]
                for k in (
                    "config",
                    "git",
                    "step",
                    "loss_tokens",
                    "input_tokens",
                    "trainable",
                    "manifest",
                    "hardware",
                    "model",
                    "training_identity",
                    "alpha_override",
                    "order_hash",
                )
            },
        )
        metric = next(
            r for r in main["rows"] if r["arm"] == arm and r["mode"] == "trained"
        )
        if budget == 250000:
            metric = read(out / "promotion.json")["splits"]["v2_validation"]["metrics"][
                arm + "/trained"
            ]
        c = state["config"]
        (published / "README.md").write_text(f"""# LoopQwen {name}

Experimental unmerged adapter on Qwen/Qwen3.5-0.8B-Base revision `{c["base_revision"]}`. See the bundled Qwen license in the adapters directory. Training uses the V2 mixed procedural corpus and seed {c["seed"]}. The six adapted layers 10–15 use rank-eight shared LoRA. The adapter has {state["trainable"]:,} trainable parameters including any learned gate. Execution uses {24 + 6 * (c["passes"] - 1)} layer calls, mode {c["mode"]} and effective alpha {metric["alpha"]}.

This development checkpoint saw {state["loss_tokens"]:,} targets in {state["step"]} updates. It scored {metric["correct"]}/{metric["n"]} final answers with {metric["verified"]} fully verified certificates on the registered V2 development slice. A single training seed limits inference about training variability. Retention NLL was {metric["retention"]:.4f} on the V2 prose sample.

Use native MLX 0.32.4 and MLX-LM 0.32.2. Reproduce with `scripts/run_phase3.sh main`{", then `scripts/run_phase3.sh promote`" if budget == 250000 else ""}. Generate with `python -m loopqwen generate --checkpoint {published} --prompt 'Evaluate ((8 + 4) * 3 - 5).' --max-tokens 96`. The prompt is raw Base-model input. For the study protocol append the traced prompt suffix from `data/v2`.
""")
    index = {
        str(p.relative_to(dest)): digest(p)
        for p in sorted(dest.rglob("*"))
        if p.is_file() and p.name != "checksums.json"
    }
    save(dest / "checksums.json", index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/measurements/phase3")
    publish(parser.parse_args().output)
