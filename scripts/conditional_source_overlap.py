"""Find exact wording shared by fresh human requests or references and any fitting source."""

import json
from collections import defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from scripts.program_language_material import (
    DEST,
    DOLLY,
    DOLLY_SHA,
    FETA,
    norm,
    source_lines,
)

RESULT = Path("docs/measurements/conditional-source-overlap.json")


def held_fields():
    original = source_lines(DOLLY, DOLLY_SHA)
    for r in read(DEST / "fresh-requests.json"):
        source = original[int(r["id"].removeprefix("dolly-"))]
        for key in ("instruction", "context", "response"):
            if source[key].strip():
                yield (r["id"], key), source[key]
    for r in read(FETA / "originals.json")["test"]:
        for key in ("question", "reference"):
            yield (r["id"], key), r[key]


def training_fields(row):
    for i, m in enumerate(
        [
            *row["messages"],
            row["target"],
            *[x["target"] for x in row.get("rejected", [])],
        ]
    ):
        for key in ("content", "reasoning_content", "tool_calls"):
            if m.get(key):
                value = m[key]
                yield (
                    f"{i}/{m['role']}/{key}",
                    value
                    if isinstance(value, str)
                    else json.dumps(value, ensure_ascii=False),
                )
    if row["tools"]:
        yield "declarations", json.dumps(row["tools"], ensure_ascii=False)


def build():
    held = {k: norm(text).split() for k, text in held_fields()}
    anchors = defaultdict(list)
    for k, words in held.items():
        assert words
        length = min(4, len(words))
        for i in range(len(words) - length + 1):
            anchors[tuple(words[i : i + length])].append((k, i))
    lengths = sorted({len(key) for key in anchors})
    best = {key: {} for key in held}
    training = DEST / "training.json.gz"
    rows = unpack(training)
    seen_words = 0
    for row in rows:
        for field, text in training_fields(row):
            words = norm(text).split()
            seen_words += len(words)
            for width in lengths:
                for i in range(len(words) - width + 1):
                    gram = tuple(words[i : i + width])
                    for key, pos in anchors.get(gram, ()):
                        target = held[key]
                        if pos and i and target[pos - 1] == words[i - 1]:
                            continue
                        n = width
                        while (
                            pos + n < len(target)
                            and i + n < len(words)
                            and target[pos + n] == words[i + n]
                        ):
                            n += 1
                        if n > best[key].get("matched_words", 0):
                            best[key] = {
                                "matched_words": n,
                                "offset": pos,
                                "training_id": row["id"],
                                "training_group": row["group"],
                                "training_field": field,
                                "phrase": " ".join(target[pos : pos + n]),
                            }
    results = []
    for (key, kind), words in held.items():
        hit = best[key, kind]
        results.append(
            {
                "id": key,
                "field": kind,
                "words": len(words),
                "minimum_anchor": min(4, len(words)),
                "whole_field": hit.get("matched_words", 0) == len(words),
                **hit,
            }
        )
    result = {
        "training_sha256": digest(training),
        "turns": len(rows),
        "training_word_occurrences": seen_words,
        "fresh_sources": len({k for k, _ in held}),
        "field_count": len(held),
        "fields": results,
    }
    save(RESULT, result)
    return result


if __name__ == "__main__":
    build()
