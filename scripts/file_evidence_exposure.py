"""Verify the exact sampled target tokens and evidence contrast from each teaching source."""

import argparse
import random
from collections import Counter
from pathlib import Path

from loopqwen.export_runtime import load_export_tokenizer
from loopqwen.io import digest, read, save
from loopqwen.research.native_function_data import token_pair
from loopqwen.research.reading_pairs import MatchedExamples


DEFAULT = Path("docs/measurements/file-evidence-exposure.json")


def exposure(root, destination=DEFAULT):
    s = read(root / "registration.json")
    train = read(s["training"])
    rows = train["examples"]
    release = Path(s["checkpoint"])
    tok = load_export_tokenizer(
        release,
        eos_token_ids=[read(release / "config.json")["text_config"]["eos_token_id"]],
    )
    bank = []
    groups = {}
    for r in rows:
        prefix, target = token_pair(tok, r)
        ident = r["id"]
        if ident.startswith("wiki-train"):
            group = "WikiSQL rehearsal"
        elif r["target"].get("tool_calls"):
            group = "file lookup"
        elif r["tools"]:
            group = "file conclusion"
            assert r["messages"][-1]["role"] == "tool"
        else:
            group = "inline conclusion"
        assert ident not in groups
        groups[ident] = group
        bank.append((ident, len(prefix) - 1, target))
    sampler = MatchedExamples(rows, bank, tok, s["pairs_per_batch"], s["batch"])
    rng = random.Random(s["fit_seed"])
    draws = Counter()
    tokens = Counter()
    checkpoints = {r["step"]: r["targets"] for r in read(root / "optimizer-steps.json")}
    used = 0
    for step in range(1, s["updates"] + 1):
        for ident, _, target in sampler.sample(rng):
            draws[ident] += 1
            tokens[groups[ident]] += len(target)
            used += len(target)
        if step in checkpoints:
            assert checkpoints[step] == used
    assert sum(draws.values()) == s["updates"] * s["batch"]
    fit = read(root / "fit.json")
    assert (
        sum(tokens.values()) == fit["targets"]
        and len(train["source_articles"]) == s["train_pairs"]
    )
    out = {
        "sha256": digest(root / "fit.json"),
        "articles": len(train["source_articles"]),
        "targets": fit["targets"],
        "updates": fit["updates"],
        "sources": {
            g: {
                "examples": sum(t == g for t in groups.values()),
                "unique_seen": sum(groups[k] == g for k in draws),
                "sampled_turns": sum(v for k, v in draws.items() if groups[k] == g),
                "target_tokens": tokens[g],
            }
            for g in sorted(set(groups.values()))
        },
    }
    if destination.exists():
        assert read(destination) == out
    else:
        save(destination, out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=Path("research/results/file-evidence-accumulated-01"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(exposure(args.study_dir, args.output))
