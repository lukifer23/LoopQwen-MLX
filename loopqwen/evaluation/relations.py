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
        clean = re.sub(
            r"^(?:final (?:answer|judgment)|answer|judgment|conclusion)\s*[:=]\s*",
            "",
            clean,
        )
        clean = clean.strip(" *`'\"")
        if clean in synonyms:
            return synonyms[clean]
        explicit = re.match(r"^(yes|no|maybe)\s*[,.:!—-]", clean)
        if explicit:
            return explicit[1]
        named = re.fullmatch(
            r"(?:therefore,? |thus,? |so,? )?(?:the (?:correct |final )?(?:answer|judgment|relation|relationship|label|claim|inference|hypothesis)(?: here)? is|it is)\s+(?:an? )?[\"'*]*(yes|no|maybe|entailment|contradiction|neutral|entailed|contradicted|uncertain|undetermined|unknown)[\"'*]*",
            clean,
        )
        if named:
            return synonyms[named[1]]
        break
    return None


def premise_randomization(differences):
    """Two-sided paired swap test, preserving all three judgments of a premise."""
    groups = [sum(differences[i : i + 3]) for i in range(0, len(differences), 3)]
    return cluster_randomization(groups)


def cluster_randomization(differences):
    """Exact two-sided paired sign randomization on integer-valued cluster totals."""
    counts = {0: 1}
    for delta in differences:
        updated = {}
        for total, n in counts.items():
            for sign in (-1, 1):
                key = total + sign * delta
                updated[key] = updated.get(key, 0) + n
        counts = updated
    observed = abs(sum(differences))
    return sum(n for d, n in counts.items() if abs(d) >= observed) / sum(
        counts.values()
    )


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
            **by,
            n=len(ix),
            wins=wins,
            losses=losses,
            p=exact_mcnemar(wins, losses),
            premise_p=premise_randomization(diff),
        )
    return result
