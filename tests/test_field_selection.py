"""Exercise the actual paired training and tool-error boundaries used by the field study."""

from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_lm.utils import load_tokenizer

from loopqwen.io import read
from loopqwen.model_loader import local_snapshot
from loopqwen.research.field_selection import positions
from loopqwen.research.native_function_data import token_pair
from scripts.field_selection_analysis import classify
from scripts.field_selection_material import TRAIN, training_views


def test_verdict_positions_index_concatenated_real_multi_length_responses():
    train = read(TRAIN)
    with TemporaryDirectory() as folder:
        views = training_views(train, Path(folder))
    table = train["tables"][0]["id"]
    selected = [w for w in views if w.get("group") == table and w["kind"] == "file"]
    assert [w["label"] for w in selected] == ["true", "false", "uncertain"]
    tok = load_tokenizer(local_snapshot())
    targets = [(None, 0, token_pair(tok, w)[1]) for w in selected]
    reason, first = positions(targets, tok)
    combined = [v for _, _, ids in targets for v in ids]
    assert [combined[j] for j in first] == [
        tok.encode(w["label"], add_special_tokens=False)[0] for w in selected
    ]
    assert all(
        tok.decode([combined[j]]).strip()
        and combined[j] != tok.convert_tokens_to_ids("</think>")
        for j in reason
    )
    assert max(reason) < len(combined) and not set(reason) & set(first)


def test_argument_reversal_is_distinct_from_a_plausible_alternate_field():
    panel = read("research/data/field-selection-panel.json")
    target = next(w["calls"][0] for w in panel["rows"] if w["family"] == "wheel")
    args = target["arguments"]

    def observed(**changes):
        return [{"call": {**target, "arguments": {**args, **changes}}}]

    assert classify(target, []) == "none"
    assert classify(target, observed()) == "exact"
    assert classify(target, observed(value_column="Release")) == "other return field"
    assert (
        classify(
            target,
            observed(key_column=args["value_column"], value_column=args["key_column"]),
        )
        == "swapped"
    )
    assert (
        classify(
            target,
            observed(
                key="another project",
                key_column=args["value_column"],
                value_column=args["key_column"],
            ),
        )
        == "other"
    )
