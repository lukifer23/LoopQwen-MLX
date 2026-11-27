"""Supervise opposite responses to a change of hypothesis or observed evidence."""

from collections import defaultdict

import mlx.core as mx


class MatchedExamples:
    def __init__(self, examples, bank, tokenizer, pairs_per_batch, batch_size):
        self.ordinary = []
        groups = defaultdict(dict)
        for row, sample in zip(examples, bank, strict=True):
            if "pair" not in row:
                self.ordinary.append(sample)
                continue
            if row["thinking"]:
                raise ValueError(
                    "Matched evidence supervision requires a shared answer prefix"
                )
            if "pair_order" in row:
                order = row["pair_order"]
            else:
                order = {"yes": 0, "no": 1}[row["visible"]]
                ids = tokenizer.encode(row["visible"], add_special_tokens=False)
                if len(ids) != 1 or sample[2][-2:-1] != ids:
                    raise ValueError("Expected an atomic binary judgment")
            group = groups[row["pair"]]
            if order not in (0, 1) or order in group:
                raise ValueError("Duplicate or invalid paired member")
            group[order] = sample
        self.pairs = [(g[0], g[1]) for g in groups.values()]
        self.count = pairs_per_batch
        self.singles = batch_size - 2 * pairs_per_batch
        if not 0 <= self.singles <= len(self.ordinary) or not 0 < self.count <= len(
            self.pairs
        ):
            raise ValueError("Unsupported paired minibatch")

    def sample(self, rng):
        selected = rng.sample(self.pairs, self.count)
        return [sample for pair in selected for sample in pair] + rng.sample(
            self.ordinary, self.singles
        )

    def targets(self, batch):
        """Locate the first genuine label divergence in each teacher-forced response."""
        positions = []
        start = 0
        for first, second in zip(
            batch[: 2 * self.count : 2], batch[1 : 2 * self.count : 2], strict=True
        ):
            a, b = first[2], second[2]
            j = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
            if j is None or j >= min(len(a), len(b)) - 1:
                raise ValueError("Expected two distinct nonterminal response labels")
            positions.append([start + j, start + len(a) + j, a[j], b[j]])
            start += len(a) + len(b)
        return mx.array(positions)

    def loss(self, logits, targets, *, individual=False):
        a, b, x, y = (targets[:, i] for i in range(4))
        positive = logits[a, x] - logits[a, y]
        negative = logits[b, x] - logits[b, y]
        if individual:
            # Both contexts must favor their own target over the other member's.
            return (
                mx.logaddexp(0, 1 - positive) + mx.logaddexp(0, 1 + negative)
            ).mean() / 2
        return mx.logaddexp(0, 1 - (positive - negative)).mean()
