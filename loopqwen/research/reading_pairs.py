"""Pairwise ranking of a positive and a negative hypothesis on the same premise."""

from collections import defaultdict

import mlx.core as mx


class MatchedExamples:
    def __init__(self, examples, bank, tokenizer, pairs_per_batch, batch_size):
        self.yes, self.no = [
            tokenizer.encode(s, add_special_tokens=False) for s in ("yes", "no")
        ]
        if len(self.yes) != 1 or len(self.no) != 1:
            raise ValueError("Expected atomic binary answer labels")
        self.tokens = mx.array([self.yes[0], self.no[0]])
        self.ordinary = []
        groups = defaultdict(dict)
        for row, sample in zip(examples, bank, strict=True):
            gold = row["visible"]
            answer = {"yes": self.yes, "no": self.no}[gold]
            if sample[2][-2:-1] != answer or row["thinking"]:
                raise ValueError("Ranking requires the unassisted shared answer prefix")
            if "pair" in row:
                group = groups[row["pair"]]
                if gold in group:
                    raise ValueError("Duplicate label in a matched premise")
                group[gold] = sample
            else:
                self.ordinary.append(sample)
        self.pairs = [(g["yes"], g["no"]) for g in groups.values()]
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

    def loss(self, full_logits, locations):
        logits = full_logits[locations[: self.count * 2]][:, self.tokens]
        margin = logits[:, 0] - logits[:, 1]
        return mx.logaddexp(0, 1 - (margin[::2] - margin[1::2])).mean()
