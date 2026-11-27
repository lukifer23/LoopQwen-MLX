"""Summarize real native charges, source exposure, retained states and pause points."""

import argparse
import json

from loopqwen.io import digest, read, save
from scripts.middle_depth_budget import BUDGET, ROOT
from scripts.middle_depth_recover import fit_directory


def retained_updates(charge, before, after):
    """Measure the verified optimizer work preserved by one resource-stopped fit."""
    from pathlib import Path

    charge = Path(charge)
    before, after = Path(before) / "state.json", Path(after) / "state.json"
    rec, start, end = read(charge), read(before), read(after)
    if (
        charge.parent != ROOT / "charges"
        or rec["stage"] != "fit"
        or rec["status"] != "stopped"
        or "Native development limit" not in rec["reason"]
        or before.parent.parent != after.parent.parent
        or start["identity"] != end["identity"]
        or start["step"] >= end["step"]
        or end["log"][: start["step"]] != start["log"]
    ):
        raise ValueError(
            "The charge and optimizer boundaries must describe a stopped continuation"
        )
    wrote = after.stat().st_mtime
    if not (
        before.stat().st_mtime
        <= rec["dispatch"]
        <= wrote
        <= rec["dispatch"] + rec["seconds"]
    ):
        raise ValueError("Checkpoint commit times are outside the native invocation")
    folder = after.parent.parent
    generations = [
        (f.stat().st_mtime, int(f.parent.name[5:]))
        for f in folder.glob("step-[0-9]*/state.json")
    ]

    def latest_at(time):
        return max(n for written, n in generations if written <= time)

    if (
        latest_at(rec["dispatch"]) != start["step"]
        or latest_at(rec["dispatch"] + rec["seconds"]) != end["step"]
        or rec["job"].split("/")[1] != folder.name
    ):
        raise ValueError("Inputs are not the actual boundary checkpoints for this fit")
    for path, state in ((before, start), (after, end)):
        if any(digest(path.parent / k) != v for k, v in state["hashes"].items()):
            raise ValueError("Unverified retained optimizer checkpoint")
    new = end["log"][start["step"] :]
    seconds = sum(r["seconds"] for r in new)
    if len(new) != end["step"] - start["step"] or seconds > rec["seconds"]:
        raise ValueError("The recorded updates do not fit inside the charged run")
    result = {
        "charge": str(charge),
        "charge_sha": digest(charge),
        "charge_seconds": rec["seconds"],
        "start_committed_at": before.stat().st_mtime,
        "end_committed_at": wrote,
        "start": str(before),
        "start_sha": digest(before),
        "end": str(after),
        "end_sha": digest(after),
        "start_step": start["step"],
        "end_step": end["step"],
        "retained_updates": len(new),
        "retained_update_seconds": seconds,
        "other_worker_seconds": rec["seconds"] - seconds,
        "method": "Sum the measured completed updates in the hash-verified continuation. The remaining worker time includes startup, checkpointing and the interrupted update. Both components are already included in the stopped charge.",
    }
    save(ROOT / "accounting" / "retained-updates" / charge.name, result)
    return result


def account():
    buckets = {
        k: {
            "ceiling": v,
            "finished": 0.0,
            "stopped": 0.0,
            "open_elapsed": 0.0,
            "attempts": 0,
            "os_disk_read_bytes": 0,
            "os_disk_write_bytes": 0,
            "peak_mlx_bytes": 0,
        }
        for k, v in BUDGET.items()
    }
    charges = {}
    for path in sorted((ROOT / "charges").glob("*.json")):
        rec = read(path)
        b = buckets[rec["stage"]]
        b["attempts"] += 1
        if rec["status"] == "running":
            b["open_elapsed"] += rec.get("observed_seconds", 0)
        else:
            charges[path.name] = digest(path)
            b["finished" if rec["status"] == "finished" else "stopped"] += rec[
                "seconds"
            ]
        first = rec.get("initial_process_disk_io")
        last = rec.get("final_process_disk_io")
        if first and last:
            b["os_disk_read_bytes"] += last["read"] - first["read"]
            b["os_disk_write_bytes"] += last["written"] - first["written"]
        b["peak_mlx_bytes"] = max(
            b["peak_mlx_bytes"],
            *(
                snap.get("peak", 0)
                for snap in [rec.get("final", {}), rec.get("last", {}), rec["initial"]]
            ),
        )
    for b in buckets.values():
        b["remaining_seconds"] = (
            b["ceiling"] - b["finished"] - b["stopped"] - b["open_elapsed"]
        )
    models = {}
    for arm in ("ordinary", "loop"):
        root = fit_directory(arm)
        latest = read(root / "latest.json")
        records = {}
        for field in ("path", "previous"):
            if not latest.get(field):
                continue
            folder = root / latest[field]
            file = folder / "state.json"
            state = read(file)
            for name, sha in state["hashes"].items():
                if digest(folder / name) != sha:
                    raise ValueError(f"Incomplete optimizer state {folder / name}")
            records[field] = {
                "path": str(folder),
                "state": digest(file),
                "weights": state["hashes"]["adapters.safetensors"],
                "step": state["step"],
                "row_exposures": sum(len(r["ids"]) for r in state["log"]),
                "update_seconds_including_features": sum(
                    r["seconds"] for r in state["log"]
                ),
                "feature_seconds": sum(r["feature_seconds"] for r in state["log"]),
            }
        assert records["path"]["state"] == latest["state"]
        models[arm] = records
    prefixes = []
    for path in sorted((ROOT / "prefix-retention").glob("*.json")):
        r = read(path)
        prefixes.append(
            {
                "manifest": str(path),
                "sha": digest(path),
                "verified_duplicate_bytes_avoided": r["avoided_duplicate_bytes"],
                "evicted": r["evicted"],
            }
        )
    return {
        "stages": buckets,
        "models": models,
        "charged_attempts": charges,
        "prefixes": prefixes,
        "accounting": "Worker occupancy includes setup and stopped attempts. Per-update times include feature work and are contained in worker time. OS process I/O counters are available on newer charge records. An open worker reports its most recent heartbeat.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot")
    p.add_argument("--retained-updates", nargs=3, metavar=("CHARGE", "BEFORE", "AFTER"))
    a = p.parse_args()
    if a.retained_updates:
        print(json.dumps(retained_updates(*a.retained_updates), indent=2))
        raise SystemExit(0)
    result = account()
    if a.snapshot:
        out = ROOT / "accounting" / (a.snapshot + ".json")
        if out.exists() and read(out) != result:
            raise FileExistsError("Name a new evidence cut")
        save(out, result)
    print(
        json.dumps(
            {
                "budgets": {
                    k: round(v["remaining_seconds"])
                    for k, v in result["stages"].items()
                },
                "latest": {k: v["path"]["step"] for k, v in result["models"].items()},
            },
            indent=2,
        )
    )
