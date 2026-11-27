"""Reconstruct teacher positions, grouped probabilities and the student's exact scheduled targets."""

import argparse
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_fit import answer_positions
from loopqwen.research.cross_view_study import NAME, tokenizer
from loopqwen.research.native_function_data import token_pair


def audit(root, destination):
    s = read(root / "registration.json")
    tok = tokenizer(s)
    examples = read(s["training"])["examples"]
    teacher = unpack(root / "teacher.json.gz")
    ts = read(root / "teacher-summary.json")
    assert (
        digest(root / "teacher.json.gz") == ts["targets_sha256"]
        and ts["teacher_weights"] == s["teacher"]["weights_sha256"]
        and ts["training_sha256"] == digest(Path(s["training"]))
    )
    assert set(teacher) == {
        r["source_question"] for r in examples if r["kind"] == "inline"
    }
    assert len(teacher) == ts["examples"] == 2 * s["train_articles"]
    article = defaultdict(dict)
    reh = []
    first = Counter()
    seen_positions = 0
    for row in examples:
        _, labels = token_pair(tok, row)
        if row["kind"] == "rehearsal":
            reh.append((row["id"], len(labels)))
            continue
        key = row["source_question"]
        q = article[row["source_article"]].setdefault(key, {})
        q[row["kind"]] = (row["id"], len(labels))
        if row["kind"] != "inline":
            continue
        tgt = teacher[key]
        positions = answer_positions(tok, labels, row["target"]["content"])
        assert tgt["positions"] == positions and tgt["labels"] == [
            labels[i] for i in positions
        ]
        for pos, ids, prob in zip(
            positions, tgt["ids"], tgt["probabilities"], strict=True
        ):
            assert len(set(ids)) == len(ids) == len(prob) == s["top_k"]
            assert (
                all(math.isfinite(v) and 0 <= v <= 1 for v in prob)
                and sum(prob) <= 1.00001
                and prob == sorted(prob, reverse=True)
            )
            assert all(0 <= i < 248320 for i in ids)
            if pos == positions[0]:
                for k in (1, 5, s["top_k"]):
                    first[k] += labels[pos] in ids[:k]
        seen_positions += len(positions)
    assert (
        seen_positions == ts["answer_positions"]
        and first[s["top_k"]] == ts["first_gold_top_k"]
        and first[s["top_k"]] / len(teacher) == ts["first_gold_top_k_rate"]
    )
    result = {
        "teacher_sha256": ts["targets_sha256"],
        "teacher_training_questions": len(teacher),
        "teacher_training_answer_positions": seen_positions,
        "first_answers_top_k": {str(k): value for k, value in first.items()},
    }
    if (root / "fit.json").is_file():
        fit = read(root / "fit.json")
        rng = random.Random(s["fit_seed"])
        groups = list(article.values())
        tokens = Counter()
        examples_seen = Counter()
        positions = 0

        def add(kind, row):
            uid, n = row
            tokens[kind] += n
            examples_seen[uid] += 1

        for _ in range(s["updates"]):
            qs = rng.choice(groups)
            opening = rng.choice(list(qs.values()))["opening"]
            other = rng.choice(reh)
            for q, v in qs.items():
                for kind in ("inline", "file"):
                    add(kind, v[kind])
                positions += len(teacher[q]["positions"]) * 2
            add("opening", opening)
            add("rehearsal", other)
        assert (
            sum(tokens.values()) == fit["targets"]
            and positions == fit["teacher_positions"]
        )
        assert sum(examples_seen.values()) == s["updates"] * 6
        result.update(
            training_targets=dict(tokens),
            training_teacher_positions=positions,
            unique_examples=len(examples_seen),
            total_examples=sum(examples_seen.values()),
            student_weights=fit["weights"],
        )
    if destination.exists():
        assert read(destination) == result
    else:
        save(destination, result)
    print(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-dir", type=Path, default=Path("research/results") / NAME)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("docs/measurements/cross-view-exposure.json"),
    )
    args = p.parse_args()
    audit(args.study_dir, args.output)
