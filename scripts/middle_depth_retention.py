"""Bound rebuildable temporal states while preserving hashed restoration recipes."""

import os
from pathlib import Path
import shutil
from time import time, monotonic

from loopqwen.io import digest, read, save
from scripts.middle_depth_budget import GIB, ROOT

WORK = Path("outputs/middle-depth")
RECIPES = ROOT / "prefix-retention"
LIMIT = 16 * GIB


def tree_size(paths):
    files = {
        f.stat().st_ino: f.stat().st_size
        for p in paths
        if p.exists()
        for f in p.rglob("*")
        if f.is_file()
    }
    return sum(files.values())


def directories():
    return list((WORK / "documents").glob("*/prefix.json")) + list(
        (WORK / "long-training").glob("*/prefix/prefix.json")
    )


def recipe(path):
    return RECIPES / (path.parent.name + "-" + path.name + ".json")


def open_prefix(path, model, ids, identity, command):
    """Reserve peak storage conservatively, including both hidden boundary files."""
    cfg = model.args
    count = sum(not model.text.layers[i].is_linear for i, p in model.sites)
    reserve = (
        len(ids)
        * (4 * cfg.num_key_value_heads * cfg.head_dim * count + 4 * cfg.hidden_size)
        + len(model.sites) * 4 * 2**20
    )
    active = [WORK / "documents", WORK / "long-training", WORK / "features"]
    cap = min(LIMIT, shutil.disk_usage(WORK).free + tree_size(active) - 24 * GIB)
    if reserve > cap:
        raise InterruptedError("A single exact document exceeds scratch headroom")
    referenced = recipe(path)
    previous = read(referenced) if referenced.exists() else None
    if previous is not None and previous["identity"] != identity:
        raise ValueError("Prefix reconstruction identity changed")
    # Keep incomplete work and unregistered legacy directories. Reclaim only
    # fully verified derivative states for which the source and command survive.
    while tree_size(active) - tree_size([path]) + reserve > cap:
        eligible = []
        for manifest in directories():
            f = manifest.parent
            r = recipe(f)
            if f == path or not r.is_file():
                continue
            receipt = read(r)
            certified = set(receipt.get("states", {})) | {"prefix.json"}
            if any(
                not file.is_file() or file.name not in certified for file in f.iterdir()
            ):
                continue
            if receipt.get("complete") and not receipt.get("evicted"):
                eligible.append((receipt["last_use"], f, r, receipt))
        if not eligible:
            raise InterruptedError(
                "Derived cache limit, no verified prefix available for rebuilding"
            )
        _, f, r, receipt = min(eligible)
        state = read(f / "prefix.json")
        for g in state["groups"]:
            for item in g["states"]:
                if digest(f / item["file"]) != item["sha256"]:
                    raise ValueError("Prefix changed before reversible reclamation")
        receipt.update(evicted=True, evicted_at=time(), former_bytes=tree_size([f]))
        save(r, receipt)
        shutil.rmtree(f)
    return previous


def seal(path, prefix, identity, command, previous):
    state = prefix.receipt
    hashes = {i["file"]: i["sha256"] for g in state["groups"] for i in g["states"]}
    if (
        previous is not None
        and previous.get("complete")
        and previous["states"] != hashes
    ):
        raise ValueError("Rebuilt complete history differs from its archived hashes")
    others = {}
    for manifest in directories():
        if manifest.parent == path:
            continue
        for group in read(manifest)["groups"]:
            for i in group["states"]:
                f = manifest.parent / i["file"]
                if f.exists():
                    others.setdefault(i["sha256"], f)
    avoided = 0
    for name, sha in hashes.items():
        f = path / name
        sibling = others.get(sha)
        if sibling is None or f.samefile(sibling):
            continue
        if digest(sibling) != sha or digest(f) != sha:
            raise ValueError("Invalid duplicate state")
        link = f.with_suffix(".verified-link")
        link.unlink(missing_ok=True)
        os.link(sibling, link)
        os.replace(link, f)
        avoided += f.stat().st_size
    save(
        recipe(path),
        dict(
            identity=identity,
            states=hashes,
            physical_sites=[
                {
                    k: v
                    for k, v in item.items()
                    if k in ("site", "kind", "consumed", "tensors")
                }
                for g in state["groups"]
                for item in g["states"]
            ],
            complete=True,
            evicted=False,
            last_use=time(),
            command=list(command),
            seconds=state["seconds"],
            avoided_duplicate_bytes=avoided
            + (previous or {}).get("avoided_duplicate_bytes", 0),
        ),
    )


def saved(model, ids, path, identity, guard, command):
    from loopqwen.research.saved_document import SavedDocument

    prev = open_prefix(path, model, ids, identity, command)
    tick = monotonic()
    ledger = RECIPES / "building" / (recipe(path).stem + ".json")
    observed = (
        read(ledger) if ledger.exists() else {"identity": identity, "attempts": []}
    )
    if observed["identity"] != identity:
        raise ValueError("Another source occupies this construction ledger")
    already_complete = (
        prev is not None and prev.get("complete") and not prev.get("evicted")
    )
    try:
        obj = SavedDocument(model, ids, path, identity, guard)
        seal(path, obj, identity, command, prev)
    except BaseException as error:
        if not already_complete:
            observed["attempts"].append(
                {
                    "seconds": monotonic() - tick,
                    "status": "stopped",
                    "error": repr(error),
                }
            )
            save(ledger, observed)
        raise
    if not already_complete:
        observed["attempts"].append(
            {"seconds": monotonic() - tick, "status": "complete"}
        )
        save(ledger, observed)
    # Existing prefixes produced before the construction ledger use the
    # sum of actual completed group timings as a separately labeled estimate.
    if (
        observed["attempts"]
        and sum(r["seconds"] for r in observed["attempts"]) >= obj.receipt["seconds"]
    ):
        last_build = max(
            i for i, r in enumerate(observed["attempts"]) if r["status"] == "complete"
        )
        earlier = [
            i
            for i, r in enumerate(observed["attempts"][:last_build])
            if r["status"] == "complete"
        ]
        first = earlier[-1] + 1 if earlier else 0
        obj.cold_seconds = sum(
            r["seconds"] for r in observed["attempts"][first : last_build + 1]
        )
        obj.cold_time_basis = (
            "wall time of construction attempts through the last complete prefix"
        )
    else:
        obj.cold_seconds = obj.receipt["seconds"]
        obj.cold_time_basis = (
            "observed completed group seconds in the saved native prefix"
        )
    return obj


def inventory():
    refs = [read(f) for f in (WORK / "features" / "inputs").glob("*.json")]
    data = {r["sha256"]: r["bytes"] for r in refs}
    for sha in data:
        if digest(WORK / "features" / "objects" / (sha + ".safetensors")) != sha:
            raise ValueError("Invalid feature object")
    save(
        ROOT / "retention.json",
        dict(
            unique_input_objects=len(data),
            input_references=len(refs),
            bytes=sum(data.values()),
            two_independent_feature_sets_bytes=2 * sum(r["bytes"] for r in refs),
            unique_working_bytes=tree_size(
                [WORK / "documents", WORK / "long-training", WORK / "features"]
            ),
            scratch_cap=LIMIT,
            free_bytes=shutil.disk_usage(WORK).free,
            prefix_manifests=len(list(RECIPES.glob("*.json"))),
            shared_donor_bytes=read(ROOT / "archive-early.json")["shared_donor_bytes"],
        ),
    )


if __name__ == "__main__":
    inventory()
