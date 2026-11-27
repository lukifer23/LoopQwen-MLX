"""Reconstruct scheduled response-token targets and compare to actual native sampler records."""

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from loopqwen.research.visit_residual_fit import annotations, stream

NAME = "visit-residual-01"
SPEC = Path("research/experiments") / (NAME + ".json")
SCHEDULE = Path("research/diagnostics/visit-residual-exposure.json")


def build(root=None):
    spec = read(SPEC)
    views = read(spec["training"])["examples"]
    tok = tokenizer(spec)
    spans = []
    for view in views:
        _, ids = token_pair(tok, view)
        body, first, contrast = annotations(tok, view, ids)
        forced = {0, first, len(ids) - 1}
        if contrast:
            forced.add(contrast[0])
        spans.append((len(ids), body, forced, bool(contrast)))
    steps = unpack(Path(root) / "updates.json.gz") if root else None
    by_group = defaultdict(Counter)
    used = defaultdict(set)
    sources = defaultdict(set)
    used_positions = defaultdict(set)
    draw = stream(views, spec["fit_seed"])
    for step in range(1, spec["updates"] + 1):
        batch = next(draw)
        rng = random.Random(f"{spec['fit_seed']}/{step}")
        if steps is not None:
            assert batch == steps[step - 1]["views"]
        for slot, i in enumerate(batch):
            n, body, fixed, contrast = spans[i]
            optional = [j for j in range(n) if j not in fixed]
            take = min(len(optional), spec["positions_per_view"] - len(fixed))
            actual = sorted(fixed.union(rng.sample(optional, take)))
            if steps is not None:
                assert steps[step - 1]["positions"][slot] == actual
            g = views[i]["group"]
            agg = by_group[g]
            agg["draws"] += 1
            agg["sampled_targets"] += len(actual)
            agg["visible_or_stop"] += sum(j >= body for j in actual)
            agg["paired_margins"] += contrast
            agg["donor_projection_rows"] += len(actual) if slot >= 4 else 0
            used[g].add(i)
            sources[g].add(views[i]["source"])
            used_positions[g].update((i, j) for j in actual)
    for group, agg in by_group.items():
        agg["unique_views"] = len(used[group])
        agg["unique_source_labels"] = len(sources[group])
        agg["unique_response_positions"] = len(used_positions[group])
    report = {
        "registration": digest(SPEC),
        "training": digest(Path(spec["training"])),
        "updates": spec["updates"],
        "per_group": dict(sorted(by_group.items())),
        "total_sampled_targets": sum(x["sampled_targets"] for x in by_group.values()),
        "donor_projection_rows": sum(
            x["donor_projection_rows"] for x in by_group.values()
        ),
    }
    if SCHEDULE.exists():
        assert read(SCHEDULE) == report
    else:
        assert root is None
        save(SCHEDULE, report)
    if root:
        fit = read(Path(root) / "fit.json")
        assert (
            len(steps) == fit["updates"]
            and fit["targets"] == report["total_sampled_targets"]
        )
        save(
            "docs/measurements/visit-residual-exposure.json",
            {
                **report,
                "observed_updates_sha256": digest(Path(root) / "updates.json.gz"),
            },
        )
    print(
        report["total_sampled_targets"],
        "candidate rows and",
        report["donor_projection_rows"],
        "reference rows",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observed", type=Path)
    build(p.parse_args().observed)
