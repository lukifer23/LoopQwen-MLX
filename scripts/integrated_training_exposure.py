"""Reconstruct semantic target positions for every scheduled source-balanced native update."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.integrated_assistance_fit import batches, positions
from loopqwen.research.native_function_data import token_pair

SPEC = Path("research/experiments/integrated-assistance-01.json")
RECEIPT = Path("research/diagnostics/integrated-assistance-exposure.json")


def build(observed=None):
    s = read(SPEC)
    tok = tokenizer(s)
    views = read(s["training"])["examples"]
    labels = []
    starts = []
    for w in views:
        _, y = token_pair(tok, w)
        start = y.index(tok.convert_tokens_to_ids("</think>")) + 1
        while start < len(y) - 1 and not tok.decode([y[start]]).strip():
            start += 1
        labels.append(y)
        starts.append(start)
    stream = batches(views, s["fit_seed"])
    logged = unpack(observed / "optimizer-steps.json.gz") if observed else None
    groups = defaultdict(Counter)
    used = defaultdict(set)
    sites = defaultdict(set)
    for step in range(1, s["updates"] + 1):
        batch = next(stream)
        if logged:
            assert logged[step - 1]["views"] == batch
        for slot, i in enumerate(batch):
            w = views[i]
            g = w["kind"] if w["group"] == "database" else w["group"]
            js = positions(
                labels[i],
                starts[i],
                s["positions_per_view"],
                f"{s['fit_seed']}/{step}/{i}",
            )
            if logged:
                assert logged[step - 1]["positions"][slot] == js
            groups[g]["draws"] += 1
            groups[g]["target_positions"] += len(js)
            groups[g]["visible_or_terminal"] += sum(j >= starts[i] for j in js)
            used[g].add(i)
            sites[g].update((i, j) for j in js)
    for g in groups:
        groups[g]["unique_views"] = len(used[g])
        groups[g]["unique_positions"] = len(sites[g])
    report = {
        "registration": digest(SPEC),
        "training": digest(Path(s["training"])),
        "groups": {g: dict(groups[g]) for g in sorted(groups)},
        "total_positions": sum(v["target_positions"] for v in groups.values()),
        "views_used": sum(len(v) for v in used.values()),
        "updates": s["updates"],
    }
    assert report["views_used"] == len(views)
    if RECEIPT.is_file():
        assert read(RECEIPT) == report
    else:
        assert observed is None
        save(RECEIPT, report)
    if observed:
        assert read(observed / "fit.json")["targets"] == report["total_positions"]
        save(
            "docs/measurements/integrated-assistance-exposure.json",
            {**report, "actual": digest(observed / "optimizer-steps.json.gz")},
        )
    print(
        report["total_positions"],
        "positions on",
        report["views_used"],
        "training turns",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observed", type=Path)
    build(p.parse_args().observed)
