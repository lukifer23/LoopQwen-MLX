"""Publish completed runs in a new directory after matching cohorts, updates and weights."""

import argparse
import shutil
import tempfile
from pathlib import Path
from statistics import median

from verify_phase6_archive import verify

from loopqwen.data.pointer6 import ROOT, SPEC
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.evaluation.order_scores import deeper, metrics, positional_lookup
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.training.mechanism import StudyArm, generation


def archive(dest, seed):
    if dest.exists():
        raise FileExistsError("Archive already exists; choose a reproduction directory")
    dest.parent.mkdir(parents=True, exist_ok=True)
    base = Path("outputs") / SPEC["id"] / str(seed)
    decision = read(base / "decision.json")
    arms = [StudyArm.registered(k, seed) for k in SPEC["arms"]]
    if decision["installed"]:
        from dataclasses import replace

        arms.append(
            replace(arms[1], name="outcome", objective="outcome", milestones=(80,))
        )
    with tempfile.TemporaryDirectory(dir=dest.parent) as td:
        root = Path(td) / "archive"
        root.mkdir()
        result = {"arms": {}, "decision": decision, "seed": seed}
        schedule = None
        order_hashes = {}
        shutil.copy("experiments/phase6.json", root / "registration.json")
        shutil.copy("experiments/phase6_followup.json", root / "conditional-registration.json")
        shutil.copy(ROOT / "manifest.json", root / "data.json")
        shutil.copy(
            "docs/measurements/phase5/base-weights-and-tokenizer.json",
            root / "base.json",
        )
        shutil.copy(
            "outputs/phase6/schedule-benchmark.json", root / "schedule-benchmark.json"
        )
        for arm in arms:
            step = arm.milestones[-1]
            gen = generation(arm, step)
            state = read(gen / "state.json")
            h = state.pop("history")
            if state["status"] != "completed" or len(h) != step:
                raise ValueError("Incomplete checkpoint")
            weights = digest(gen / "adapters.safetensors")
            path = root / arm.name
            path.mkdir()
            seq = [(r["depth"], r["ids"]) for r in h]
            if arm.name != "outcome":
                if schedule is None:
                    schedule = seq
                    if seed == 109 and fingerprint(seq) != fingerprint(
                        unpack(Path("docs/measurements/phase5/schedule.json.gz"))
                    ):
                        raise ValueError("Historical epoch schedule changed")
                    packed(root / "schedule.json.gz", schedule)
                if seq != schedule:
                    raise ValueError(
                        "Mismatched training maps, boundaries or curriculum"
                    )
                orders = [r["order_digest"] for r in h]
                if order_hashes.setdefault(arm.rendering, orders) != orders:
                    raise ValueError("Mismatched training presentations")
            seconds = sum(r["seconds"] for r in h)
            maps = sum(r["maps"] for r in h)
            labels = sum(r["labels"] for r in h)
            info = {
                "updates": step,
                "maps": maps,
                "unique": len({i for _, group in seq for i in group}),
                "labels": labels,
                "seconds": seconds,
                "maps_s": maps / seconds,
                "labels_s": labels / seconds,
                "median_ms": median(r["seconds"] for r in h) * 1000,
                "peak_gib": max(r["peak"] for r in h) / 2**30,
                "min_available_gib": min(r["available"] for r in h) / 2**30,
                "positions": sum(r["positions"] for r in h),
                "site_positions": sum(r["site_positions"] for r in h),
                "capacity": state["capacity"],
                "weights": weights,
                "optimizer": digest(gen / "optimizer.safetensors"),
                "evaluations": [],
            }
            packed(path / "updates.json.gz", h)
            save(path / "state.json", state)
            for n in arm.milestones:
                source = arm.output / f"orders-{n}.json.gz"
                measured = unpack(source)
                if measured["identity"]["weights"] != digest(
                    generation(arm, n) / "adapters.safetensors"
                ):
                    raise ValueError("Incorrect evaluated milestone")
                shutil.copy(source, path / source.name)
                scores = metrics(measured["records"])
                if n == arm.milestones[-1]:
                    info["positional_lookup"] = positional_lookup(measured["records"])
                info["evaluations"].append(
                    {
                        "path": str((path / source.name).relative_to(root)),
                        "step": n,
                        "scores": scores,
                    }
                )
            if arm.kind == "recurrent":
                shutil.copy(gen / "adapters.safetensors", path / "adapters.safetensors")
                deep = unpack(arm.output / "deep.json.gz")
                if deep["identity"]["weights"] != weights:
                    raise ValueError("Incorrect extrapolation checkpoint")
                shutil.copy(arm.output / "deep.json.gz", path / "deep.json.gz")
                info["deep"] = f"{arm.name}/deep.json.gz"
                save(path / "depth-summary.json", deeper(deep["records"]))
            for key, src in (
                ("causal", f"causal-{step}.json.gz"),
                ("latency", f"latency-{step}.json"),
                ("late", f"late-{step}.json.gz"),
                ("benchmark", "benchmark.json"),
            ):
                if (arm.output / src).exists():
                    shutil.copy(arm.output / src, path / src)
                    info[key] = f"{arm.name}/{src}"
            required = {"latency"}
            if arm.name == "randomized":
                required.update(("late", "benchmark"))
            if decision["installed"] and arm.name in ("randomized", "outcome"):
                required.add("causal")
            if required - info.keys():
                raise ValueError(
                    f"Missing measurements for {arm.name}: {required - info.keys()}"
                )
            if arm.kind == "recurrent":
                cond = info["evaluations"][-1]["scores"]["conditions"]
                rates = ", ".join(f"{v:.1%}" for v in cond["mean_random"]["accuracy"])
                ms = read(root / info["latency"])["results"]
                latency = next(
                    r["median_ms"] for r in ms if r["batch"] == 1 and r["depth"] == 3
                )
                command = (
                    "scripts/run_phase6.sh train --arms " + arm.name
                    if arm.name != "outcome"
                    else "scripts/run_phase6.sh followup"
                )
                (path / "README.md").write_text(
                    f"# {arm.name.capitalize()} categorical pointer adapter, seed {seed}\n\n"
                    f"Qwen/Qwen3.5-0.8B-Base at `{SPEC['base_revision']}` with {state['capacity']:,} trainable weights. "
                    "Rank-eight LoRA adapts the shared layers 10–15. A shared 8,192-weight head reads the final prompt position. "
                    "The artifact requires the base weights (Apache 2.0) and this repository's native runtime (MLX 0.32.4 and MLX-LM 0.32.2).\n\n"
                    f"This stage applied {step} AdamW updates, {maps:,} map exposures and {labels:,} state labels under the {arm.objective} objective. "
                    f"The training renderer was {arm.rendering}. Parent adapter SHA-256: `{state['parent_weights']}`. "
                    f"The data manifest fingerprint is `{fingerprint(read(root / 'data.json'))}`. "
                    f"The adapter and categorical head SHA-256 is `{weights}`.\n\n"
                    f"Development accuracies on eight shuffled orders of 384 unseen permutation maps at hops 1–3 were {rates}. "
                    f"Median three-hop interactive latency was {latency:.1f} ms. Depths 4–6 are uncached extrapolation. "
                    "These measurements cover eight-state permutation maps and one training seed. "
                    "For language generation and retention see the separate research report.\n\n"
                    f"```sh\n{command} --seed {seed}\n.venv/bin/python scripts/phase6_predict.py --arm {arm.name} --verify\n"
                    f".venv/bin/python scripts/phase6_predict.py --arm {arm.name} --map 5 4 7 6 0 2 1 3 --start 0 --depth 3\n```\n"
                )
            result["arms"][arm.name] = info
        save(root / "results.json", result)
        save(
            root / "checksums.json",
            {
                str(p.relative_to(root)): digest(p)
                for p in sorted(root.rglob("*"))
                if p.is_file()
            },
        )
        verify(root)
        root.rename(dest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=109)
    p.add_argument("--destination", type=Path, default=Path("docs/measurements/phase6"))
    args = p.parse_args()
    archive(args.destination, args.seed)
