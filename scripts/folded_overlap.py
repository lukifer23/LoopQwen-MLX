"""Compare actual fitting tokens with the human content of every newly enrolled request."""

import csv
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from mlx_lm.utils import load_tokenizer

from loopqwen.io import read
from loopqwen.model_loader import local_snapshot
from loopqwen.research.registered import registered


def words(text):
    return [
        "#" if re.search(r"\d", w) else w for w in re.findall(r"\w+", text.casefold())
    ]


def grams(ws):
    return {tuple(ws[i : i + 5]) for i in range(len(ws) - 4)}


def audit(train, panel, path):
    strings = [words(text) for _, text in train]
    bank = [grams(tokens) for tokens in strings]
    owners = defaultdict(list)
    for j, features in enumerate(bank):
        for feature in features:
            owners[feature].append(j)
    rows = []
    for w in panel:
        clean = (
            w["question"]
            .removesuffix("\nGive the final numeric answer.")
            .removesuffix("\nAnswer yes or no based on the passage.")
            .removeprefix("Passage\n")
            .replace("\n\nQuestion\n", "\n")
        )
        tokens = words(clean)
        features = grams(tokens)
        overlap = defaultdict(int)
        for key in features:
            for j in owners[key]:
                overlap[j] += 1
        nearest = max(
            overlap,
            key=lambda j: overlap[j] / (len(features) + len(bank[j]) - overlap[j]),
            default=None,
        )
        jaccard = (
            overlap[nearest] / (len(features) + len(bank[nearest]) - overlap[nearest])
            if nearest is not None
            else 0
        )
        longest = max(
            (
                SequenceMatcher(None, tokens, strings[j], autojunk=False)
                .find_longest_match()
                .size
                for j in overlap
            ),
            default=0,
        )
        rows.append(
            [
                w["id"],
                w["source"],
                len(features),
                len(overlap),
                jaccard,
                longest if longest >= 5 else "",
                train[nearest][0] if nearest is not None else "",
            ]
        )
    with path.open("w") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "id",
                "fresh_source",
                "distinct_quantity_masked_fivegrams",
                "training_contexts_sharing_fivegram",
                "maximum_jaccard",
                "longest_contiguous_shared_run_if_at_least_five",
                "closest_training_id",
            ]
        )
        writer.writerows(rows)
    print(
        path.name,
        len(rows),
        "sharing 5-gram",
        sum(r[3] > 0 for r in rows),
        "highest Jaccard",
        max(r[4] for r in rows),
        "longest",
        max(r[5] or 0 for r in rows),
    )


def fresh(spec):
    return [
        r for r in read(spec["material"])["rows"] if r["family"] in ("judgment", "math")
    ]


def main():
    folded = registered("folded-system-01")
    tok = load_tokenizer(local_snapshot())
    fitted = [
        (r["id"], tok.decode(r["ids"], skip_special_tokens=True))
        for r in read(folded["training"])["rows"]
        if r["split"] == "train"
    ]
    audit(fitted, fresh(folded), Path("paper/folded-training-overlap.csv"))

    shared = registered("shared-session-01")
    training = read(shared["training"])
    typed_spec = read(Path(shared["public"]) / "registration.json")
    typed = read(typed_spec["training"])["examples"]

    # Include answer supervision and earlier assistant/tool turns. Shared system
    # prompt scaffolding can dilute Jaccard, so we omit only the system role.
    def context(r):
        parts = [m["content"] for m in r["messages"] if m["role"] != "system"]
        return "\n".join([*parts, r["thinking"], r["visible"]])

    corpora = [(r["source"] + ":" + r["id"], context(r)) for r in training["examples"]]
    corpora.extend(
        ("routing:" + r["id"], r["question"])
        for r in training["router"]
        if r["split"] == "train"
    )
    corpora.extend(("typed:" + str(i), context(r)) for i, r in enumerate(typed))
    audit(corpora, fresh(shared), Path("paper/shared-training-overlap.csv"))


if __name__ == "__main__":
    main()
