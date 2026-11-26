"""Fit a second scope decision on the user's actual question, keeping all task weights fixed."""

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn

from ..context_policy import features
from ..identity import fingerprint
from ..io import save


def train(s, t, model, rows, folder, budget):
    positives = sorted({r["question"] for r in rows})
    subjects = sorted({q.rsplit(" ", 1)[1][:-1] for q in positives})
    negatives = {
        "How many are there?",
        "Which animal occurs twice?",
        "Who is gray?",
        "Which names appear in this list?",
        "What colors appear in the passage?",
        "What color occurs most often?",
    }
    for subject in subjects:
        negatives.update(
            pattern.format(subject)
            for pattern in (
                "What animal is {}?",
                "What species is {}?",
                "Is {} a frog?",
                "Who is in the same class as {}?",
                "Where is {}?",
                "Is {} here?",
                "Is {} mentioned?",
                "How many times does {} appear?",
                "Can you identify {}'s animal type?",
                "What does {} fear?",
                "What color is different from {}'s?",
                "Is {} white?",
            )
        )
    negatives = sorted(negatives)
    texts = positives + negatives
    ids = [t.encode(q, add_special_tokens=False) for q in texts]
    vectors = mx.stack([features(model.text.model.embed_tokens, x) for x in ids])
    mx.eval(vectors)
    y = mx.array([1] * len(positives) + [0] * len(negatives))
    mx.random.seed(s["seed"])
    head = nn.Linear(vectors.shape[-1], 2)
    opt = optim.Adam(0.003)

    def loss(net):
        both = nn.losses.cross_entropy(net(vectors), y, reduction="none")
        return 0.5 * (both[: len(positives)].mean() + both[len(positives) :].mean())

    update = nn.value_and_grad(head, loss)
    for _ in range(320):
        budget.check()
        val, grad = update(head)
        opt.update(head, grad)
        mx.eval(val, head.parameters(), opt.state)
    head.save_weights(str(folder / "intent.safetensors"))
    save(
        folder / "intent.json",
        {
            "confidence": 0.99,
            "seed": s["seed"],
            "positive": len(positives),
            "negative": len(negatives),
            "training_questions": fingerprint(texts),
            "training_ids": fingerprint(ids),
        },
    )
