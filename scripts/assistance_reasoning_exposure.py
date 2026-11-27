"""Count actual thinking-content gradient targets separately from Qwen framing and visible text."""

from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from scripts.integrated_training_exposure import SPEC

DEST = Path("docs/measurements/integrated-assistance-reasoning-exposure.json")


def observed(root):
    spec = read(SPEC)
    tok = tokenizer(spec)
    close = tok.convert_tokens_to_ids("</think>")
    terminal = tok.convert_tokens_to_ids("<|im_end|>")
    examples = read(spec["training"])["examples"]
    tokens = []
    source = defaultdict(Counter)
    for w in examples:
        group = w["kind"] if w["group"] == "database" else w["group"]
        _, y = token_pair(tok, w)
        j = y.index(close)
        assert y[-1] == terminal
        # Rendered reasoning can include boundary newlines. Fast-tokenizer offsets
        # keep UTF-8 fragments and punctuation while excluding whitespace-only tokens.
        raw = tok.decode(y[:j])
        text = w["target"].get("reasoning_content", "").strip()
        assert raw.strip() == text
        encoded = tok._tokenizer(
            raw, add_special_tokens=False, return_offsets_mapping=True
        )
        assert encoded["input_ids"] == y[:j]
        body = {
            i for i, (a, b) in enumerate(encoded["offset_mapping"]) if raw[a:b].strip()
        }
        visible = next(
            (i for i in range(j + 1, len(y) - 1) if tok.decode([y[i]]).strip()),
            len(y) - 1,
        )
        kinds = [
            "thinking_text"
            if i in body
            else "framing"
            if i < visible
            else "end_of_turn"
            if i == len(y) - 1
            else "visible_text"
            for i in range(len(y))
        ]
        tokens.append((group, kinds))
        source[group]["unique_turns"] += 1
        source[group]["nonempty_thought_turns"] += bool(text)
        source[group]["available_thinking_text_positions"] += len(body)
    groups = defaultdict(Counter)
    seen = defaultdict(set)
    log = unpack(root / "optimizer-steps.json.gz")
    for step in log:
        for index, js in zip(step["views"], step["positions"], strict=True):
            g, kinds = tokens[index]
            counts = Counter(kinds[j] for j in js)
            groups[g].update(counts)
            groups[g]["training_turns"] += 1
            groups[g]["draws_with_thinking_target"] += bool(counts["thinking_text"])
            seen[g].update((index, j) for j in js if kinds[j] == "thinking_text")
    overall = Counter()
    records = {}
    for g, count in groups.items():
        rec = dict(count + source[g])
        rec["distinct_thinking_targets_selected"] = len(seen[g])
        records[g] = rec
        overall.update(rec)
    assert (
        sum(
            overall[k]
            for k in ("thinking_text", "framing", "end_of_turn", "visible_text")
        )
        == read(root / "fit.json")["targets"]
    )
    assert overall["training_turns"] == len(log) * 7 and overall["unique_turns"] == len(
        examples
    )
    return {
        "registration": digest(SPEC),
        "draws_sha256": digest(root / "optimizer-steps.json.gz"),
        "token_boundary": "Thinking text means a token wholly or partly overlapping a non-whitespace character before the thinking close marker. Framing includes the close marker and surrounding whitespace. The end-of-turn ID is counted separately.",
        "groups": records,
        "totals": dict(overall),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = observed(args.root)
    if DEST.exists():
        assert result == read(DEST)
    else:
        save(DEST, result)
    print(result["totals"])
