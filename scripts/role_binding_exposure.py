"""Recount which original training examples and response targets entered the optimizer."""

import random
from collections import Counter, defaultdict
from pathlib import Path

from mlx_lm.utils import load_tokenizer

from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.research.native_function_data import token_pair
from loopqwen.research.reading_pairs import MatchedExamples

ROOT = Path(__file__).resolve().parents[1]
NAME = "role-binding-01"
DEST = ROOT / "docs/measurements/role-binding-training-exposure.json"


def run():
    folder = ROOT / "research/results" / NAME
    spec = read(folder / "registration.json")
    fit = read(folder / "fit.json")
    examples = read(ROOT / spec["training"])["examples"]
    tok = load_tokenizer(local_snapshot())
    bank = [(r["id"], 0, token_pair(tok, r)[1]) for r in examples]
    sampler = MatchedExamples(
        examples, bank, tok, spec["pairs_per_batch"], spec["batch"]
    )
    groups = {}
    paired = defaultdict(list)
    for r in examples:
        if "pair" in r:
            g = "table call" if r["target"].get("tool_calls") else "meaning"
            paired[r["pair"]].append(r)
        elif r["id"].startswith("wiki-train-"):
            g = "table return"
        else:
            g = "rehearsal"
        assert r["id"] not in groups
        groups[r["id"]] = g
    draws = Counter()
    tokens = Counter()
    directions = Counter()
    for pair in paired.values():
        a, b = sorted(pair, key=lambda r: r["pair_order"])
        if groups[a["id"]] != "table call":
            continue
        x, y = [r["target"]["tool_calls"][0]["function"]["arguments"] for r in (a, b)]
        assert x.keys() == y.keys() and x["document"] == y["document"]
        directions[next(k for k in x if x[k] != y[k])] += 1
    rng = random.Random(spec["fit_seed"])
    for _ in range(spec["updates"]):
        for ident, _, ids in sampler.sample(rng):
            draws[ident] += 1
            tokens[groups[ident]] += len(ids)
    assert (
        sum(draws.values()) == spec["batch"] * spec["updates"]
        and sum(tokens.values()) == fit["targets"]
    )
    counts = {
        g: {
            "available": sum(v == g for v in groups.values()),
            "unique_seen": sum(groups[k] == g for k in draws),
            "draws": sum(n for k, n in draws.items() if groups[k] == g),
            "target_tokens": tokens[g],
        }
        for g in sorted(set(groups.values()))
    }
    result = {
        "fit_sha256": digest(folder / "fit.json"),
        "training_sha256": fit["training"],
        "optimizer_updates": spec["updates"],
        "target_tokens": sum(tokens.values()),
        "by_training_source": counts,
        "first_different_table_parameter": dict(directions),
    }
    if DEST.exists():
        assert read(DEST) == result
    else:
        save(DEST, result)
    print(result)


if __name__ == "__main__":
    run()
