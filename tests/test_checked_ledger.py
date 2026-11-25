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
        records = re.findall(r"^([a-z]+ [a-z]+ [a-z]+) = (\d{6})$", rendered, re.M)
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
