"""Score the explicit final relation in a completed visible answer, independently of case."""

import re

from .statistics import exact_mcnemar


def conclusion(visible):
    synonyms = {
        "yes": "yes",
        "entailment": "yes",
        "entailed": "yes",
        "no": "no",
        "contradiction": "no",
        "contradicted": "no",
        "maybe": "maybe",
        "neutral": "maybe",
        "uncertain": "maybe",
        "undetermined": "maybe",
        "unknown": "maybe",
    }
    for line in reversed(visible.splitlines()):
        if not line.strip():
            continue
        clean = line.strip(" \t*#`.!").lower()
        if clean in synonyms:
            return synonyms[clean]
        m = re.match(
            r"^(?:final (?:answer|judgment)|answer|judgment|conclusion)\s*[:=]\s*(.*)",
            clean,
        )
        if m:
            clean = m[1]
        words = re.findall(r"\b(?:" + "|".join(synonyms) + r")\b", clean)
        if words and len({synonyms[v] for v in words}) == 1:
            return synonyms[words[0]]
        break
    return None


def summarize(items, records):
    result = {}
    for name in ("all", "esnli", "multinli"):
        ix = [i for i, r in enumerate(items) if name == "all" or r["source"] == name]
        by = {}
        for arm in ("candidate", "base"):
            selected = [records[arm][i] for i in ix]
            by[arm] = {
                "correct": sum(r["correct"] for r in selected),
                "complete": sum(r["complete"] for r in selected),
                "probe": sum(
                    r["probe"] == items[i]["label"]
                    for i, r in zip(ix, selected, strict=True)
                ),
                "seconds": sum(r["seconds"] for r in selected),
                "tokens": sum(len(r["token_ids"]) for r in selected),
                "triplets": sum(
                    all(r["correct"] for r in selected[j : j + 3])
                    for j in range(0, len(selected), 3)
                ),
            }
        diff = [
            int(records["candidate"][i]["correct"]) - int(records["base"][i]["correct"])
            for i in ix
        ]
        wins, losses = diff.count(1), diff.count(-1)
        result[name] = dict(
            **by, n=len(ix), wins=wins, losses=losses, p=exact_mcnemar(wins, losses)
        )
    return result
