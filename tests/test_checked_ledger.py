import re

from mlx_lm.utils import load_tokenizer

from loopqwen.model_loader import local_snapshot
from loopqwen.research.checked_rationale import checked_response
from loopqwen.research.ledger import ledger


def test_inverse_check_uses_publisher_operands_and_exact_gold():
    assert "0.2 * 60 = 12" in checked_response("Rate <<12/60=0.2>>0.2.\n#### 0.2")
    assert "-7 + 2 = -5" in checked_response("Total <<-5-2=-7>>-7.\n#### -7")
    assert checked_response("Total <<2*0=0>>0.\n#### 0") is None
    assert checked_response("Total <<3+2=4>>4.\n#### 4") is None
    assert checked_response("Total <<3+2=5>>5.\n#### 4") is None


def test_frozen_ledger_labels_and_positions_are_in_the_encoded_input():
    from loopqwen.io import read

    t = load_tokenizer(local_snapshot())
    for frozen in read("research/data/ledger-curriculum-01.json")["items"]:
        row = ledger(
            t,
            frozen["seed"],
            frozen["minimum_length"],
            frozen["fraction"],
            frozen["announce"],
        )
        assert frozen == row
        rendered = t.decode(row["prompt_ids"])
        records = re.findall(
            r"^([a-z]+ [a-z]+ [a-z]+) = (\d{6})$", rendered, re.MULTILINE
        )
        assert (
            len(records)
            == len(dict(records))
            == len({v for _, v in records})
            == row["entries"]
        )
        assert records[int(len(records) * row["fraction"])] == (row["key"], row["gold"])
        assert rendered.count(row["gold"]) == 1
        assert (f"Retrieve the value for {row['key']}." in rendered) == row["announce"]
        assert rendered.endswith("<think>\n")


def test_split_preference_gradient_matches_autodiff():
    import mlx.core as mx
    import numpy as np

    from loopqwen.research.paired_solutions import preference_terms

    baseline = mx.array([85.0, 55.0])
    beta = 0.1

    def objective(nll):
        logratio = baseline - nll
        return mx.logaddexp(mx.array(0.0), -beta * (logratio[0] - logratio[1]))

    for values in ([85.0, 55.0], [101.0, 48.0], [60.0, 69.0]):
        true_loss, true_gradient = mx.value_and_grad(objective)(mx.array(values))
        loss, slope = preference_terms(*values, baseline.tolist(), beta)
        np.testing.assert_allclose(loss, true_loss.item(), rtol=1e-6)
        np.testing.assert_allclose([slope, -slope], true_gradient.tolist(), rtol=1e-6)


def test_preference_selection_requires_comparable_scored_answers():
    from loopqwen.research.paired_solutions import preference

    def draw(n, answer):
        return {"token_ids": [0] * n, "correct": answer == "4", "answer": answer}

    rows = [draw(55, None), draw(60, "3"), draw(19, "4")]
    assert preference(rows) is None
    rows.append(draw(41, "4"))
    assert preference(rows) == (3, 1)
