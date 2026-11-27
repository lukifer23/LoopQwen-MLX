"""Score file and calendar grounding against the real tools offered to the models."""

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from loopqwen.io import read
from loopqwen.research.valid_return import score
from scripts.valid_return_material import tool_set


def test_grounded_calendar_decision_requires_both_real_returns_in_order(tmp_path):
    panel = read("research/data/valid-return-panel.json")
    item = next(q for q in panel["rows"] if q["id"].endswith("-weekday"))
    (tmp_path / item["document"]).write_text(panel["documents"][item["document"]])
    tools = {t.name: t for t in tool_set(item, tmp_path)}
    refs = item["ordered_events"]

    def execute(call):
        return {
            "call": call,
            "observation": tools[call["name"]].callback(**call["arguments"]),
        }

    date, weekday = [execute(e["call"]) for e in refs]
    assert [date, weekday] == refs
    # Both alternatives succeed on the exact same real event and date.
    wrong_field = execute(
        {
            "name": "csv_lookup",
            "arguments": {**date["call"]["arguments"], "value_column": "Room"},
        }
    )
    wrong_operation = execute(
        {
            "name": "day_of_year",
            "arguments": weekday["call"]["arguments"],
        }
    )
    assert wrong_field["observation"]["value"] != date["observation"]["value"]
    assert wrong_operation["observation"]["value"] != weekday["observation"]["value"]

    tok = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[UNK]": 0, "</think>": 1}, unk_token="[UNK]")
        )
    )

    def result(*events, final=None):
        trajectory = {
            "status": "answered",
            "answer": final or item["gold"],
            "turns": [*events, {"token_ids": [1]}],
        }
        return score(tok, item, 0, trajectory)

    for sequence in (
        [],
        [date],
        [weekday],
        [weekday, date],
        [wrong_field, weekday],
        [date, wrong_operation],
    ):
        measured = result(*sequence)
        assert measured["correct_answer"] and not measured["success"]
    assert result(date, weekday)["success"]
    assert result(wrong_field, date, weekday)["success"]
    assert not result(date, weekday, final="uncertain")["success"]
