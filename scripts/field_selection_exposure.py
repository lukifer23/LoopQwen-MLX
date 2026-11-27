"""Independently reconstruct class order, real source draws and optimizer exposure."""

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair

LABELS = ("true", "false", "uncertain")


def run(root):
    spec = read(root / "registration.json")
    tok = tokenizer(spec)
    ids = {k: tok.encode(k, add_special_tokens=False)[0] for k in LABELS}
    close = tok.convert_tokens_to_ids("</think>")
    views = unpack(root / "training-views.json.gz")
    store = defaultdict(list)
    groups = {}
    for w in views:
        _, target = token_pair(tok, w)
        kind = w["kind"]
        entry = {
            "id": w["id"],
            "length": len(target),
            "group": w.get("group"),
            "kind": kind,
            "reason": 0,
        }
        if kind in ("inline", "file"):
            p = target.index(close)
            gold = next(t for t in target[p + 1 : -1] if tok.decode([t]).strip())
            assert gold == ids[w["label"]]
            entry["reason"] = sum(bool(tok.decode([t]).strip()) for t in target[:p])
            groups.setdefault(w["group"], {}).setdefault(kind, {})[w["label"]] = entry
        store[kind].append(entry)
    ordered = list(groups)
    assert len(ordered) == 128 and all(
        set(table) == {"inline", "file"}
        and all(set(q) == set(LABELS) for q in table.values())
        for table in groups.values()
    )
    seed = random.Random(spec["fit_seed"])
    progress = []
    exposure = Counter()
    by_view = Counter()
    distinct = defaultdict(set)
    counts = Counter()
    targets = 0
    reasons = 0
    for i in range(1, spec["updates"] + 1):
        group = seed.choice(ordered)
        view = seed.choice(("inline", "file"))
        kind = seed.choice(("opening", "direction", "repair"))
        package = seed.choice(("package_open", "package_answer"))
        batch = [groups[group][view][k] for k in LABELS] + [
            seed.choice(store[kind]),
            seed.choice(store[package]),
            seed.choice(store["rehearsal"]),
        ]
        by_view[view] += 1
        for e in batch:
            key = e["kind"]
            counts["triplet" if key in ("inline", "file") else key] += 1
            exposure[key] += e["length"]
            distinct[key].add(e["id"])
            targets += e["length"]
            reasons += e["reason"]
        if i % 100 == 0:
            progress.append(
                {"step": i, "targets": targets, "explanation_targets": reasons}
            )
    fit = read(root / "fit.json")
    assert (
        targets == fit["targets"]
        and reasons == fit["explanation_targets"]
        and dict(counts) == fit["draws"]
    )
    snapshots = read(root / "optimizer-steps.json")
    assert progress == [{k: w[k] for k in progress[0]} for w in snapshots]
    output = {
        "registration": digest(root / "registration.json"),
        "examples": len(views),
        "targets": targets,
        "human_field_tokens": reasons,
        "response_draws": dict(counts),
        "source_tokens": dict(exposure),
        "unique_views_exposed": {k: len(v) for k, v in distinct.items()},
        "triplet_steps_by_view": dict(by_view),
        "steps": progress,
    }
    destination = Path("docs/measurements/field-selection-training-exposure.json")
    if destination.exists():
        assert read(destination) == output
    else:
        save(destination, output)
    print(
        sum(counts.values()),
        "draws,",
        targets,
        "targets,",
        reasons,
        "human field tokens,",
        len(snapshots),
        "optimizer snapshots",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root", type=Path, default=Path("research/results/field-selection-01")
    )
    run(p.parse_args().root)
