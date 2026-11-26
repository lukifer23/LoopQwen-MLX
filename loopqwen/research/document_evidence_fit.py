"""Read learned first-pass passage evidence from the actual serialized search observation."""

import json
from pathlib import Path

from ..io import digest, read
from ..tool_session import compact, prompt
from .dialogue_state_fit import train
from .tool_fit import restore


def passage_ends(tok, row):
    known = row.get("known", {})
    if not known:
        return [], []
    message = row["messages"][-1]
    assert message["role"] == "tool"
    passages = json.loads(message["content"])["passages"]
    assert len(known) == len(passages) and list(known) == [
        str(i + 1) for i in range(len(passages))
    ]
    text = prompt(tok, row["messages"])
    start = text.rindex(message["content"])
    offsets = tok._tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True
    )
    assert offsets["input_ids"] == tok.encode(text, add_special_tokens=False)
    positions = []
    for passage in passages:
        # A causal key at the end of each serialized excerpt has seen its entire body.
        fragment = compact(passage)
        end = text.index(fragment, start) + len(fragment)
        positions.append(
            next(
                i for i, (a, b) in enumerate(offsets["offset_mapping"]) if a < end <= b
            )
        )
        start = end
    return positions, list(known.values())


def fit(spec, root, budget):
    parent = Path(spec["initial_adapter"])
    assert (
        digest(parent / "checksums.json") == spec["initial_sha256"]
        and read(parent / "decision.json")["passed"]
    )
    model, tok = restore(parent)
    train(
        model,
        tok,
        spec,
        root,
        budget,
        locator=passage_ends,
        kinds=("search", "read", "replay"),
    )


def validation(root, spec):
    data = read(spec["training"])
    fit = read(root / "fit.json")
    assert (
        fit["training"] == digest(Path(spec["training"]))
        and fit["weights"] == digest(root / "tool-adapters.safetensors")
        and fit["head"] == digest(root / "state-head.safetensors")
        and fit["passes"] == 2
        and fit["turns"] == len(data["examples"])
        and fit["updates"] == spec["updates"]
    )
    results = read(root / "validation.json")
    counts = {False: [0, 0], True: [0, 0]}
    for row, res in zip(data["validation"], results, strict=True):
        assert (
            row["id"] == res["id"]
            and row["known"] == res["known"]
            and row["known"].keys() == res["probabilities"].keys()
        )
        for key, y in row["known"].items():
            p = res["probabilities"][key]
            assert 0 <= p <= 1
            counts[y][0] += int((p >= 0.5) == y)
            counts[y][1] += 1
    return {"positive": counts[True], "negative": counts[False]}
