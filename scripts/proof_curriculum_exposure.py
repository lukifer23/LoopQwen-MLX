"""Reconstruct the optimizer's exact example and target-token exposure at every saved update."""

import argparse
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
    views = unpack(root / "training-views.json.gz")
    bank = defaultdict(lambda: defaultdict(dict))
    rehearsal = []
    for w in views:
        _, target = token_pair(tok, w)
        reason = 0
        if w["kind"] in ("inline", "file"):
            boundary = target.index(close)
            reason = sum(bool(tok.decode([i]).strip()) for i in target[:boundary])
        item = (
            w["id"],
            len(target),
            reason,
            w.get("case", w["id"]),
            w.get("source", "table"),
            w["kind"],
        )
        if w["kind"] == "rehearsal":
            rehearsal.append(item)
        else:
            bank[w["source"]][w["label"]].setdefault(w["case"], {})[w["kind"]] = item
    seed = random.Random(spec["fit_seed"])
    labels = ("true", "false", "uncertain")
    seen = defaultdict(set)
    visited = Counter()
    tokens = Counter()
    proofs = Counter()
    samples = Counter()
    progress = []

    def draw(source, choices):
        label = seed.choice(choices)
        return seed.choice(list(bank[source][label].values()))

    for step in range(1, spec["updates"] + 1):
        other_source = "package" if seed.random() < 0.25 else "proof"
        h = draw("human", labels)
        e = draw(other_source, labels[:2] if other_source == "package" else labels)
        opening = seed.choice([h, e])["opening"]
        old = seed.choice(rehearsal)
        for uid, n, proof, case, src, kind in (
            h["inline"],
            h["file"],
            e["inline"],
            e["file"],
            opening,
            old,
        ):
            key = src + "." + kind
            seen[src].add(case)
            samples[key] += 1
            visited[uid] += 1
            tokens[key] += n
            proofs[key] += proof
        if step % 100 == 0:
            progress.append(
                {
                    "step": step,
                    "targets": sum(tokens.values()),
                    "explanation_targets": sum(proofs.values()),
                }
            )
    assert (
        sum(tokens.values()) == fit["targets"]
        and sum(proofs.values()) == fit["explanation_targets"]
    )
    assert len(views) == fit["turns"]
    snapshots = read(root / "optimizer-steps.json")
    assert progress == [
        {k: r[k] for k in ("step", "targets", "explanation_targets")} for r in snapshots
    ]
    result = {
        "views": digest(root / "training-views.json.gz"),
        "fit": digest(root / "fit.json"),
        "sampled_turns": sum(samples.values()),
        "unique_turns": len(visited),
        "unique_sources": {k: len(v) for k, v in seen.items()},
        "source_draws": dict(samples),
        "target_tokens": dict(tokens),
        "reasoning_targets": dict(proofs),
        "progress": progress,
    }
    path = Path("docs/measurements/proof-curriculum-training-exposure.json")
    if path.exists():
        assert read(path) == result
    else:
        save(path, result)
    print(
        "Reconstructed",
        result["sampled_turns"],
        "turns and",
        sum(tokens.values()),
        "targets including",
        sum(proofs.values()),
        "reasoning targets.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--study-dir", type=Path, default=Path("research/results/proof-curriculum-01")
    )
    run(p.parse_args().study_dir)
