"""Recover a complete AdamW generation if interruption split rename and pointer update."""

import re
from pathlib import Path

from loopqwen.io import digest, read, save


def reconcile(folder):
    folder = Path(folder)
    pointer = folder / "latest.json"
    current = read(pointer) if pointer.is_file() else None

    def check(path):
        s = read(path / "state.json")
        if (
            path.name != f"step-{s['step']:04d}"
            or s["next_rng_seed"] != s["identity"]["seed"] + s["step"]
        ):
            raise ValueError("Invalid trained checkpoint generation")
        if len(s["log"]) != s["step"] or [r["step"] for r in s["log"]] != list(
            range(1, s["step"] + 1)
        ):
            raise ValueError("Incomplete optimizer history")
        for key, sha in s["hashes"].items():
            if digest(path / key) != sha:
                raise ValueError("Incomplete persisted checkpoint")
        if (
            read(path / "registration.json")["weights"]
            != s["hashes"]["adapters.safetensors"]
        ):
            raise ValueError("Checkpoint inference weights differ from training state")
        return s

    if current:
        p = folder / current["path"]
        if digest(p / "state.json") != current["state"]:
            raise ValueError("Broken latest pointer")
        last = check(p)
    else:
        last = None
    for p in sorted(folder.glob("step-*")):
        if not re.fullmatch(r"step-\d{4,}", p.name):
            continue
        if last and int(p.name[5:]) <= last["step"]:
            continue
        nxt = check(p)
        if last and (
            nxt["identity"] != last["identity"]
            or nxt["log"][: last["step"]] != last["log"]
        ):
            raise ValueError(
                "A completed unreferenced generation diverges from the saved lineage"
            )
        current = dict(
            path=p.name,
            previous=current["path"] if current else None,
            state=digest(p / "state.json"),
        )
        save(pointer, current)
        last = nxt
    return current


if __name__ == "__main__":
    import argparse
    from loopqwen.worker import acquire

    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    acquire()
    reconcile(p.parse_args().directory)
