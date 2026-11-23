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
    dest.mkdir(exist_ok=True, parents=True)
    metrics = {}
    index = {}
    for cfg in sorted((out / "configs").glob("*.yaml")):
        root = Path("checkpoints") / cfg.stem
        if not (root / "latest.json").is_file():
            continue
        gen = root / read(root / "latest.json")["directory"]
        state = read(gen / "state.json")
        metrics[cfg.stem] = performance(state)
        index[cfg.stem] = dict(
            git=state["git"],
            config=state["config"],
            dataset=state["sampling"],
            source=state.get("training_identity"),
            status=state.get("status"),
            generation=gen.name,
            metadata=digest(gen / "state.json"),
            adapters=digest(gen / "adapters.safetensors"),
        )
        json_gzip(dest / "curves" / (cfg.stem + ".json.gz"), state["history"])
    save(dest / "training.json", metrics)
    save(dest / "registry.json", index)
    for name in (
        "screen.json",
        "main.json",
        "selection.json",
        "learn1-lr.json",
        "learn2-lr.json",
        "fresh-development.json",
        "native-audit.json",
        "v3-manifest.json",
        "retention-manifest.json",
        "export-audit.json",
        "compaction.json",
        "fork-audit.json",
    ):
        if (out / name).is_file():
            shutil.copyfile(out / name, dest / name)
    for name in ("frontier.json",):
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
    for arm in ("dense", "direct", selection["selected"]):
        name = "p3_" + arm
        root = Path("checkpoints") / name
        gen = root / read(root / "targets-100000.json")["directory"]
        published = Path("adapters/phase3") / name
        published.mkdir(exist_ok=True, parents=True)
        state = read(gen / "state.json")
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
                    "training_identity",
                    "alpha_override",
                    "order_hash",
                )
            },
        )
        metric = next(
            r for r in main["rows"] if r["arm"] == arm and r["mode"] == "trained"
        )
        c = state["config"]
        (published / "README.md").write_text(f"""# LoopQwen {name}

Experimental unmerged adapter on Qwen/Qwen3.5-0.8B-Base revision `{c["base_revision"]}`. See the bundled Qwen license in the adapters directory. Training uses the V2 mixed procedural corpus and seed {c["seed"]}. The six adapted layers 10–15 use rank-eight shared LoRA. The adapter has {state["trainable"]:,} trainable parameters including any learned gate. Execution uses {24 + 6 * (c["passes"] - 1)} layer calls, mode {c["mode"]} and effective alpha {metric["alpha"]}.

This development checkpoint saw {state["loss_tokens"]:,} targets in {state["step"]} updates. It scored {metric["correct"]}/{metric["n"]} final answers with {metric["verified"]} fully verified certificates on the registered V2 development slice. A single training seed limits inference about training variability. Retention NLL was {metric["retention"]:.4f} on the V2 prose sample.

Use native MLX 0.32.4 and MLX-LM 0.32.2. Reproduce with `scripts/run_phase3.sh main`. Generate with `python -m loopqwen generate --checkpoint {published} --prompt 'Evaluate ((8 + 4) * 3 - 5).' --max-tokens 96`. The prompt is raw Base-model input. For the study protocol append the traced prompt suffix from `data/v2`.
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
