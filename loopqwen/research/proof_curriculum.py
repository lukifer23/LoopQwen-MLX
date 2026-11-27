"""Train the repeated physical visit to improve the supported verdict after evidence."""

from collections import Counter
from functools import partial
from pathlib import Path

import mlx.core as mx
from mlx import nn

from ..io import digest, read, save
from . import reasoned_evidence as comparison
from .reasoned_evidence_fit import LABELS, fit, supervised_turns

NAME = "proof-curriculum-01"


def turns(spec, root):
    data = read(spec["training"])
    return supervised_turns(
        [(k, w) for k in ("human", "proof", "package") for w in data[k]]
    )


def qualification(spec):
    data = read(spec["training"])
    proof = data["proof"]
    counts = Counter(r["label"] for r in proof)
    return {
        "training": digest(Path(spec["training"])),
        "material": digest(Path(spec["material"])),
        "distinct_worlds": len({r["theory"] for r in proof}),
        "verdicts": dict(counts),
        "human": len(data["human"]),
        "packages": len({r["package"] for r in data["package"]}),
        "version_checks": len(data["package"]),
        "multihop_known": sum(
            r["label"] != "uncertain" and r["depth"] >= 2 for r in proof
        ),
    }


def qualify(spec, tok, root, guard):
    from scripts.proof_curriculum_material import build

    build()
    guard.check()
    report = qualification(spec)
    counts = report["verdicts"]
    save(root / "source-qualification.json", report)
    if not (
        report["distinct_worlds"] == 192
        and all(counts[k] == 64 for k in LABELS)
        and report["multihop_known"] >= 24
        and report["packages"] >= 8
        and report["human"] == 384
    ):
        raise InterruptedError(
            "Published logic, depth or real distribution source coverage missed its minimum"
        )


def draw(rng, streams, rehearsal):
    source = "package" if rng.random() < 0.25 else "proof"

    def example(k, labels):
        return rng.choice(streams[k][rng.choice(labels)])

    pair = [
        example("human", LABELS),
        example(source, LABELS if source == "proof" else LABELS[:2]),
    ]
    return pair, rng.choice(pair)["opening"], rng.choice(rehearsal)


def refinement(tok):
    classes = mx.array([tok.encode(k, add_special_tokens=False)[0] for k in LABELS])
    assert len(set(classes.tolist())) == len(LABELS)

    def objective(model, hidden, rows, cols, target, reasoning, answer):
        visits = model.sites[model.span.start :]
        assert visits == ((23, 0), (23, 1))
        first = model._execute(hidden, visits[:1])
        second = model._execute(first, visits[1:])

        def project(h):
            return model.project(model.text.model.norm(h)).astype(mx.float32)

        logits = project(second[rows, cols])
        ce = nn.losses.cross_entropy(logits, target, reduction="none")
        starts = answer[mx.array([0, answer.size // 2])]
        reference = mx.stop_gradient(
            project(first[rows[starts], cols[starts]])[:, classes]
        )
        active = logits[starts][:, classes]
        gold = target[starts, None] == classes[None, :]

        def odds(scores):
            return mx.sum(mx.where(gold, scores, 0), axis=-1) - mx.logsumexp(
                mx.where(gold, -mx.inf, scores), axis=-1
            )

        gap = mx.maximum(0, 0.25 + odds(reference) - odds(active)).mean()
        supervised = (
            0.50 * ce[reasoning].mean() + 0.35 * ce[answer].mean() + 0.15 * ce.mean()
        )
        return 0.85 * supervised + 0.15 * gap

    return objective


def verify(root, decision):
    comparison.verify(root, decision, name=NAME, make_views=turns)
    s = read(root / "registration.json")
    source = read(root / "source-qualification.json")
    assert source == qualification(s)


if __name__ == "__main__":
    comparison.run(
        name=NAME,
        qualify=qualify,
        trainer=partial(fit, make_views=turns, draw_cases=draw, refinement=refinement),
        extra_sources=(
            "loopqwen/research/proof_curriculum.py",
            "scripts/proof_curriculum_material.py",
        ),
    )
