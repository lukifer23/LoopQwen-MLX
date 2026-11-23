import random
from fractions import Fraction

import pytest

from loopqwen.data.reasoning import FAMILIES, problem
from loopqwen.evaluation.phase2 import numeric


@pytest.mark.parametrize(
    "text,gold",
    [
        ("3*2=6\nFinal answer: 6e-1", "3/5"),
        (".500", "1/2"),
        ("#### 3/6", "1/2"),
        ("Final answer: 3\nFinal answer: 4", None),
        ("There are 3 boats", None),
        ("Final answer: 2\n3", None),
        ("1/0", None),
    ],
)
def test_numeric(text, gold):
    assert numeric(text) == gold


def test_gold_and_fingerprints():
    random_source = random.Random(123)
    for family in FAMILIES:
        for level in range(3):
            for ood in (True, False):
                for _ in range(20):
                    r = problem(random_source, family, level, ood)
                    assert numeric(r["trace"]) == str(Fraction(r["answer"]))


def test_v2_targets():
    from loopqwen.data.tokenize import batch
    from loopqwen.data.v2 import rows

    docs = rows("validation", "trace")[:3]
    x, pos, targets = batch(docs)
    start = 0
    for lane, doc in enumerate(docs):
        n = doc["output_length"]
        first = lane * x.shape[1] + doc["prefix"] - 1
        assert pos[start : start + n].tolist() == list(range(first, first + n))
        assert targets[start : start + n].tolist() == doc["ids"][doc["prefix"] :]
        start += n
    assert start == targets.size
