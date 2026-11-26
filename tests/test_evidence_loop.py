from loopqwen.io import read
from loopqwen.research.deduction_data import answer
from loopqwen.research.science_data import normalize


def test_disjoint_memory_sources_and_frozen_gold():
    first = read("research/data/deduction-stories-01.json")
    second = read("research/data/deduction-stories-02.json")
    historical = {
        r["story"] for split, rows in first.items() if split != "train" for r in rows
    }
    seen = set()
    for split, rows in second.items():
        for r in rows:
            assert r["story"] not in seen
            seen.add(r["story"])
            if split != "train":
                assert r["story"] not in historical
            assert len(set(r["supports"])) == 2
    assert answer("She is afraid of wolves.") == "wolf"
    assert answer("cats or sheep") is None


def test_passage_uses_only_a_finished_user_message_and_open_assistant_turn():
    from mlx_lm.utils import load_tokenizer

    from loopqwen.evidence_loop import passage
    from loopqwen.model_loader import local_snapshot

    t = load_tokenizer(local_snapshot())
    row = read("research/data/deduction-stories-02.json")["replication"][0]
    assert passage(t, row["prompt_ids"]) == (row["facts"], row["question"])
    assert passage(t, row["prompt_ids"] + [248069]) is None
    assert passage(t, row["prompt_ids"] + [248046, 248045, 846, 198]) is None


def test_science_supports_and_distractors_do_not_cross_splits():
    table = read("research/data/science-composition-01.json")
    seen = set()
    for rows in table.values():
        current = {
            normalize(text) for r in rows for text in [*r["facts"], r["question"]]
        }
        assert current.isdisjoint(seen)
        assert all(len(set(r["supports"])) == 2 and len(r["facts"]) == 6 for r in rows)
        seen.update(current)
