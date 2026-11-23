"""Serial native research stages with immutable resolved configurations."""

import argparse
import os
from pathlib import Path
import sys
import subprocess
from statistics import median
from time import perf_counter

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loopqwen.io import read, save, digest, checkpoint_location
from scripts.phase2_supervise import supervise

OUT = Path("outputs/phase4")
SPEC = read("experiments/phase4.json")


def configuration(arm, seed):
    return {
        **SPEC["config"],
        **SPEC["arms"][arm],
        "name": f"p4_{seed}_{arm}",
        "seed": seed,
        "milestones": SPEC["milestones"],
    }


def run_train(cfg):
    path = OUT / "configs" / (cfg["name"] + ".yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and yaml.safe_load(path.read_text()) != cfg:
        raise ValueError(f"Existing configuration differs: {path}")
    path.write_text(yaml.safe_dump(cfg))
    root = Path("checkpoints") / cfg["name"]
    journal = OUT / "runs" / (cfg["name"] + ".json")
    if journal.exists() and read(journal)["status"] == "completed":
        st = read(checkpoint_location(root) / "state.json")
        if (
            st["status"] != "completed"
            or digest(checkpoint_location(root) / "adapters.safetensors")
            != read(journal)["weights"]
        ):
            raise ValueError("Recorded checkpoint changed")
        return st
    start = perf_counter()
    save(journal, dict(status="started", config=digest(path)))
    try:
        subprocess.run(
            [sys.executable, "-m", "loopqwen", "train", "--config", str(path)],
            check=True,
        )
    except subprocess.CalledProcessError as ex:
        save(
            journal,
            dict(
                status="interrupted" if ex.returncode in (-2, -15, 130) else "failed",
                code=ex.returncode,
            ),
        )
        raise
    gen = checkpoint_location(root)
    st = read(gen / "state.json")
    save(
        journal,
        dict(
            status=st["status"],
            seconds=perf_counter() - start,
            weights=digest(gen / "adapters.safetensors"),
            config=digest(path),
        ),
    )
    return st


def efficiency(seed):
    results = []
    for cap, pol in [(None, "off")] + [
        (cap, pol) for cap in (256, 360, 512, 768) for pol in ("off", "block", "all")
    ]:
        c = configuration("gated", seed)
        c.update(
            name=f"p4_profile_{cap}_{pol}",
            steps=10,
            eval_count=4,
            checkpoint_interval=10,
            milestones=[],
            gradient_checkpointing=pol != "off",
            checkpoint_policy="block" if pol == "block" else "all",
        )
        if cap is None:
            c.pop("max_microbatch_positions")
        else:
            c["max_microbatch_positions"] = cap
        s = run_train(c)
        h = s["history"][2:]
        sec = sum(r["seconds"] for r in h)
        results.append(
            dict(
                cap=cap,
                policy=pol,
                order=s["order_hash"],
                tps=sum(r["labels"] for r in h) / sec,
                inputs_s=sum(r["inputs"] for r in h) / sec,
                positions_s=sum(r["executed_positions"] for r in h) / sec,
                update_times=[r["seconds"] for r in h],
                median_s=median(r["seconds"] for r in h),
                peak=max(r["peak_bytes"] for r in h),
                available=min(r["available"] for r in h),
                cached=max(r["mlx_cached"] for r in h),
                active=max(r["mlx_active"] for r in h),
            )
        )
        save(OUT / "profiles.json", results)
    if len({r["order"] for r in results}) != 1:
        raise ValueError("Unmatched profile workloads")
    eligible = [
        r
        for r in results
        if r["cap"] and r["peak"] < 6 * 2**30 and r["available"] > 1.5 * 2**30
    ]
    if not eligible:
        raise MemoryError("No profile reached the registered memory envelope")
    selection = max(eligible, key=lambda r: r["tps"])
    save(
        OUT / "execution.json",
        dict(
            max_microbatch_positions=selection["cap"],
            gradient_checkpointing=selection["policy"] != "off",
            checkpoint_policy="block" if selection["policy"] == "block" else "all",
        ),
    )


def stage(name, seed):
    if digest(Path("data/v2/manifest.json")) != SPEC["data_manifest"]:
        raise ValueError("Unregistered V2 manifest")
    if name == "audit":
        subprocess.run(
            [sys.executable, "-m", "loopqwen.benchmark.phase4_gradients"], check=True
        )
        return
    if name == "efficiency":
        efficiency(seed)
        return
    policy = read(OUT / "execution.json")

    def arm(key):
        return configuration(key, seed) | policy

    if name == "replicate":
        gated = run_train(arm("gated") | {"label_budget": SPEC["primary_targets"]})
        seconds = sum(r["seconds"] for r in gated["history"])
        cfg = arm("dense") | {"seconds_budget": seconds}
        dense = run_train(cfg)
        root = Path("checkpoints") / cfg["name"]
        marker = root / "targets-250000.json"
        if not marker.exists():
            dense = run_train(
                arm("dense")
                | {
                    "name": f"p4_{seed}_dense_data",
                    "label_budget": SPEC["primary_targets"],
                }
            )
        else:
            dense = read(root / read(marker)["directory"] / "state.json")
        if (dense["order_hash"], dense["loss_tokens"]) != (
            gated["order_hash"],
            gated["loss_tokens"],
        ):
            raise ValueError("Unmatched experimental cohorts")
    elif name == "dual-exit":
        run_train(arm("joint") | {"label_budget": SPEC["dual_targets"]})
    elif name == "latent":
        subprocess.run(
            [sys.executable, "-m", "loopqwen.training.latent", "--seed", str(seed)],
            check=True,
        )
    elif name == "verify":
        for args in (
            ["-m", "loopqwen.benchmark.phase4_gradients"],
            ["-m", "loopqwen.benchmark.phase4_gradients", "--bf16"],
            ["scripts/phase3_regression.py", "--output", "outputs/phase4"],
        ):
            subprocess.run([sys.executable, *args], check=True)
        root = Path("checkpoints") / f"p4_{seed}_gated"
        gen = root / read(root / "targets-250000.json")["directory"]
        dest = Path("exports") / f"p4_stock_{seed}"
        if not dest.exists():
            subprocess.run(
                [sys.executable, "-m", "loopqwen.stock_export", str(gen), str(dest)],
                check=True,
            )
        deployment = read(dest / "deployment.json")
        if deployment["source"] != digest(gen / "adapters.safetensors"):
            raise ValueError("Stock export source mismatch")
        save(OUT / "deployment.json", deployment)
    elif name in ("boundary", "v4"):
        if not (OUT / f"evaluation-{seed}.json").is_file():
            raise ValueError(
                "Score the primary experiments before capacity and data ablations"
            )
        if name == "v4":
            subprocess.run([sys.executable, "-m", "loopqwen.data.v4"], check=True)
        for key in ("dense", "joint"):
            for weighted in (False, True) if name == "v4" else (False,):
                cfg = arm(key)
                cfg.update(
                    name=f"p4_{seed}_{key}_{name}" + ("_weighted" if weighted else ""),
                    label_budget=25000,
                )
                if name == "boundary":
                    cfg["target"] = "boundary"
                else:
                    cfg["dataset"] = "v4"
                if weighted:
                    cfg["token_weights"] = {
                        "final": 2.0,
                        "format": 0.25,
                        "intermediate": 1.0,
                        "eos": 0.5,
                        "language": 1.0,
                    }
                run_train(cfg)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "loopqwen.evaluation.phase4",
                "--seed",
                str(seed),
                "--suite",
                name,
            ],
            check=True,
        )
    elif name == "evaluate":
        subprocess.run(
            [sys.executable, "-m", "loopqwen.evaluation.phase4", "--seed", str(seed)],
            check=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stages",
        nargs="+",
        choices=[
            "audit",
            "efficiency",
            "replicate",
            "dual-exit",
            "latent",
            "evaluate",
            "v4",
            "boundary",
            "verify",
        ],
    )
    p.add_argument("--seed", type=int, default=SPEC["seed"])
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    lock = OUT / "study.lock"
    if not a.worker:
        sys.exit(supervise([sys.executable, __file__, *sys.argv[1:], "--worker"], lock))
    fd = int(os.environ["LOOPQWEN_STUDY_FD"])
    subprocess.run(
        [sys.executable, "scripts/phase2_supervise.py", "--check", str(lock)],
        check=True,
        pass_fds=(fd,),
    )
    for key in a.stages:
        stage(key, a.seed)
