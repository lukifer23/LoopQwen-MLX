"""Publish a complete pointer study into a new directory, verifying each checkpoint and prediction."""

import argparse
import gzip
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from statistics import median
from datetime import datetime, timezone

from verify_phase5_archive import verify
from phase5_extrapolation import summarize, training_rows
from phase5_selection import decide


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if path.suffix == ".gz":
        raw = gzip.compress(raw, mtime=0)
    path.write_bytes(raw)


def archive(dest, seed):
    spec = read(Path("experiments/phase5.json"))
    data = Path("data/phase5_pointer")
    out = Path("outputs/phase5")
    results = out / str(seed)
    if dest.exists():
        raise FileExistsError(f"{dest} exists; choose a new reproduction directory")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as temp:
        root = Path(temp) / "archive"
        root.mkdir()
        shutil.copy(data / "manifest.json", root / "dataset.json")
        shutil.copy("experiments/phase5.json", root / "registration.json")
        write(
            root / "cases.json.gz",
            [json.loads(s) for s in (data / "cases.jsonl").read_bytes().splitlines()],
        )
        write(root / "profile.json.gz", read(out / "profile.json"))
        runs = {}
        schedule = None
        prior = read(data / "manifest.json")["prior"]

        def passes(arm):
            result = read(results / f"{arm}-480.json")
            return all(
                m["accuracy"]
                >= max(p["conditional"], p["prior"], p["majority"])
                + spec["prior_margin"]
                for m, p in zip(result["metrics"], prior, strict=True)
            ) and result["metrics"][2]["correct"] > sum(
                x["pred"][0] == x["gold"][2] for x in result["records"]
            )

        arms = list(spec["arms"])
        selected = "recurrent"
        if not passes(selected):
            selected = "bridge"
            arms.append(selected)
        if passes(selected):
            arms.extend(["multi", "terminal", "outcome"])
        for arm in arms:
            checkpoint = Path("checkpoints") / f"p5_{seed}_{arm}"
            final = 560 if arm == "outcome" else 480
            gen = checkpoint / read(checkpoint / f"at-{final}.json")["directory"]
            state = read(gen / "state.json")
            h = state.pop("history")
            if state["status"] != "completed" or state["step"] != final:
                raise ValueError(f"Incomplete {arm}")
            matched = [[r["depth"], r["ids"]] for r in h]
            if len(h) != (80 if arm == "outcome" else 480) or any(
                len(x[1]) != 8 for x in matched
            ):
                raise ValueError("Incomplete exposure")
            if arm != "outcome":
                if schedule is not None and schedule != matched:
                    raise ValueError("Different logical schedules")
                schedule = matched
            for step in [560] if arm == "outcome" else spec["milestones"]:
                r = read(results / f"{arm}-{step}.json")
                g = checkpoint / read(checkpoint / f"at-{step}.json")["directory"]
                if (
                    r["training"] != state["identity"]
                    or r["weights"] != sha(g / "adapters.safetensors")
                    or r["optimizer"] != sha(g / "optimizer.safetensors")
                ):
                    raise ValueError("Stale milestone evidence")
                write(root / f"predictions/{arm}-{step}.json.gz", r)
            weights = sha(gen / "adapters.safetensors")
            if arm in set(spec["arms"]) | {"bridge"}:
                timing = read(results / f"latency-{arm}.json")
                if timing["identity"]["weights"] != weights:
                    raise ValueError("Timing used different checkpoint weights")
                write(root / f"latency/{arm}.json", timing)
            if arm in ("recurrent", "bridge", "outcome"):
                folder = root / "models" / arm
                folder.mkdir(parents=True)
                shutil.copy(
                    gen / "adapters.safetensors", folder / "adapters.safetensors"
                )
                write(folder / "state.json", state)
            if arm in ("recurrent", "bridge", "outcome"):
                d = read(results / f"diagnostics-{arm}-{final}.json")
                if d["weights"] != weights or d["cases"] != sha(data / "cases.jsonl"):
                    raise ValueError("Stale diagnostic evidence")
                write(root / f"predictions/diagnostics-{arm}.json.gz", d)
            if passes(selected) and arm in (selected, "middle", "full", "outcome"):
                extra = read(results / f"extra-{arm}.json")
                if extra["weights"] != weights:
                    raise ValueError("Stale extrapolation")
                write(root / f"predictions/extra-{arm}.json.gz", extra)
                write(
                    root / f"extrapolation/{arm}.json",
                    summarize(extra, training_rows()),
                )
            write(root / f"history/{arm}.json.gz", h)
            seconds = sum(r["seconds"] for r in h)
            runs[arm] = dict(
                provenance=state,
                weights=weights,
                optimizer=r["optimizer"],
                updates=len(h),
                maps=sum(r["maps"] for r in h),
                unique=len({i for _, ids in matched for i in ids}),
                labels=sum(r["labels"] for r in h),
                intermediate=sum(r["intermediate"] for r in h),
                positions=sum(r["positions"] for r in h),
                site_positions=sum(r["site_positions"] for r in h),
                block_applications=sum(
                    r["maps"]
                    * (
                        r["depth"]
                        if arm in ("recurrent", "bridge", "terminal", "outcome")
                        else 1
                    )
                    for r in h
                ),
                update_seconds=seconds,
                median_seconds=median(x["seconds"] for x in h),
                peak_bytes=max(x["peak"] for x in h),
                min_available=min(x["available"] for x in h),
                metrics=r["metrics"],
                runtime=r["runtime"],
            )
        stage_times = {}
        for name in ("profile", "install", "followup"):
            log = out / (name + ".log")
            if log.is_file():
                stat = log.stat()
                if stat.st_size and hasattr(stat, "st_birthtime"):
                    stage_times[name] = dict(
                        first_log_creation_utc=datetime.fromtimestamp(
                            stat.st_birthtime, timezone.utc
                        ).isoformat(),
                        last_log_write_utc=datetime.fromtimestamp(
                            stat.st_mtime, timezone.utc
                        ).isoformat(),
                        elapsed_through_last_log_seconds=stat.st_mtime
                        - stat.st_birthtime,
                    )
        write(root / "stage-times.json", stage_times)
        write(root / "training.json", runs)
        write(root / "schedule.json.gz", schedule)
        write(root / "decision.json", decide(root))
        write(
            root / "checksums.json",
            {
                str(p.relative_to(root)): sha(p)
                for p in sorted(root.rglob("*"))
                if p.is_file()
            },
        )
        verify(root)
        root.rename(dest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, default=Path("docs/measurements/phase5"))
    p.add_argument("--seed", type=int, default=109)
    args = p.parse_args()
    archive(args.destination, args.seed)
