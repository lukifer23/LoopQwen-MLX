import random
from fractions import Fraction

import pytest

from loopqwen.data.reasoning import FAMILIES, problem
from loopqwen.evaluation.answers import numeric


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


def test_exact_pairing_and_vote():
    from loopqwen.evaluation.statistics import exact_mcnemar, paired
    from loopqwen.evaluation.inference_budget import choose

    assert exact_mcnemar(0, 5) == 0.0625
    assert exact_mcnemar(3, 3) == 1.0
    with pytest.raises(ValueError):
        paired([{"id": "A", "correct": True}], [{"id": "B", "correct": True}])
    assert (
        choose([{"prediction": "5"}, {"prediction": "6"}, {"prediction": "6"}]) == "6"
    )
    assert (
        choose([{"prediction": "5"}, {"prediction": "6"}, {"prediction": None}]) == "5"
    )


def test_recorded_split_and_order():
    import json
    from loopqwen.data.v2 import ROOT
    from loopqwen.data.sampling import prepare
    from loopqwen.config import config

    def ids(split):
        return [
            json.loads(line)["id"]
            for line in (ROOT / (split + ".jsonl")).read_text().splitlines()
        ]

    all_ids = sum(
        (
            ids(split)
            for split in (
                "train",
                "validation",
                "test",
                "ood",
                "language_train",
                "language_validation",
                "language_test",
            )
        ),
        [],
    )
    assert len(set(all_ids)) == len(all_ids)
    a = config("configs/phase2/p2_a_trace_loop.yaml")
    r, _, _, pol = prepare(a)
    dense, _, _, other = prepare(a | {"passes": 1})
    assert pol == other and pol["excluded"] == {}
    assert [s["id"] for s in r] == [s["id"] for s in dense]


def test_cosine_schedule_and_budget_validation(tmp_path):
    import mlx.core as mx
    import yaml
    from loopqwen.config import config
    from loopqwen.training.optimizers import optimizer

    values = {
        "name": "test_schedule",
        "steps": 10,
        "schedule": "cosine",
        "warmup_fraction": 0.2,
        "min_lr": 1e-5,
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(values))
    c = config(path)
    o = optimizer(c)
    observed = []
    for i in range(10):
        o.apply_gradients({"w": mx.array(1.0)}, {"w": mx.array(1.0)})
        observed.append(o.learning_rate.item())
    assert observed[0] == pytest.approx(c["learning_rate"] / 2)
    assert max(observed) == pytest.approx(c["learning_rate"])
    assert observed[-1] == pytest.approx(c["min_lr"])
    path.write_text(yaml.safe_dump(values | {"label_budget": 0}))
    with pytest.raises(ValueError):
        config(path)
