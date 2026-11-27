"""Independently recover preferences, real repair draws and both response exposures at every checkpoint."""

import argparse
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair


def run(root):
    spec = read(root / "registration.json")
    fit = read(root / "fit.json")
    tok = tokenizer(spec)
    close = tok.convert_tokens_to_ids("</think>")
    rows = unpack(root / "training-views.json.gz")
    odds = unpack(root / "reference-logodds.json.gz")
    by = defaultdict(lambda: defaultdict(dict))
    for w in rows:
        prefix, response = token_pair(tok, w)
        reason = 0
        if w["kind"] == "verdict":
            cut = response.index(close)
            reason = sum(bool(tok.decode([v]).strip()) for v in response[:cut])
        record = (w["id"], len(response), reason, prefix, response)
        by[w["kind"]][w["source"]].setdefault(w["pair"], {})[w["choice"]] = record
    assert set(odds) == {
        k
        for kind, sub in by.items()
        if kind in ("field", "operation", "verdict")
        for gr in sub.values()
        for k in gr
    }
    assert all(math.isfinite(v) for v in odds.values())
    for sub in by.values():
        for candidates in sub.values():
            for record in candidates.values():
                if "rejected" in record:
                    pos, neg = record["preferred"], record["rejected"]
                    assert pos[3] == neg[3] and pos[4] != neg[4]
    seed = random.Random(spec["fit_seed"])
    counts = Counter()
    distinct = defaultdict(set)
    tokens = neg_tokens = reason = 0
    points = []
    for step in range(1, spec["updates"] + 1):
        action = "operation" if seed.random() < 0.18 else "field"
        for kind in (action, "verdict", "repair", "rehearsal"):
            source = seed.choice(list(by[kind]))
            group = by[kind][source]
            key = seed.choice(list(group))
            pairs = group[key]
            who = kind + "." + source
            counts[who] += 1
            distinct[who].add(key)
            tokens += pairs["preferred"][1]
            reason += pairs["preferred"][2]
            if "rejected" in pairs:
                neg_tokens += pairs["rejected"][1]
        if step % 100 == 0:
            points.append(
                {
                    "step": step,
                    "targets": tokens,
                    "explanation_targets": reason,
                    "rejected_tokens": neg_tokens,
                }
            )
    assert (tokens, reason, neg_tokens, dict(counts)) == (
        fit["targets"],
        fit["explanation_targets"],
        fit["rejected_tokens"],
        fit["draws"],
    )
    assert points == [
        {k: x[k] for k in points[0]} for x in read(root / "optimizer-steps.json")
    ]
    receipt = {
        "registration": digest(root / "registration.json"),
        "reference": digest(root / "reference-logodds.json.gz"),
        "reference_pairs": len(odds),
        "response_views": len(rows),
        "preferred_draws": sum(counts.values()),
        "preferred_tokens": tokens,
        "rejected_tokens": neg_tokens,
        "explanation_tokens": reason,
        "steps": points,
        "draws_by_source": dict(counts),
        "distinct_by_source": {k: len(v) for k, v in distinct.items()},
    }
    path = Path("docs/measurements/valid-return-exposure.json")
    if path.exists():
        assert receipt == read(path)
    else:
        save(path, receipt)
    print(
        len(odds),
        "complete paired reference odds;",
        tokens,
        "preferred targets,",
        neg_tokens,
        "rejected targets",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root", type=Path, default=Path("research/results/valid-return-01")
    )
    run(p.parse_args().root)
