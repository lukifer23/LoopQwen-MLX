"""Audit reuse of text independently of tool schemas, chat framing and model outcomes."""

import re
from collections import Counter, defaultdict
from pathlib import Path
from unicodedata import normalize

from loopqwen.io import digest, read, save
from scripts.assistance_deliberation import write_csv

ROOT = Path("research/sources")


def words(text):
    return tuple(re.findall(r"\w+", normalize("NFKC", text).casefold()))


def grams(tokens):
    if len(tokens) < 4:
        return {tokens}
    return {tokens[i : i + 4] for i in range(len(tokens) - 3)}


def nearest(kind, train, held):
    # Index every shared four-word substring. A source without a shared substring has score zero.
    references, postings = {}, defaultdict(set)
    originals = defaultdict(list)
    for index, text in train:
        normalized = words(text)
        originals[normalized].append(index)
        group = grams(normalized)
        references[index] = group
        for gram in group:
            postings[gram].add(index)
    result = []
    for name, family, text in held:
        normalized = words(text)
        group = grams(normalized)
        counts = Counter(k for v in group for k in postings[v])
        scores = [
            (overlap / (len(references[k]) + len(group) - overlap), k)
            for k, overlap in counts.items()
        ]
        high, key = max(scores, default=(0, ""))
        result.append(
            {
                "audit": kind,
                "id": name,
                "family": family,
                "words": len(normalized),
                "normalized_exact_matches": len(originals[normalized]),
                "nearest_fourgram_jaccard": high,
                "nearest_training_id": key,
            }
        )
    return result


def build():
    s = read("research/experiments/integrated-assistance-01.json")
    training = read(s["training"])["examples"]
    panel = read(s["material"])["rows"]
    # Distinct literal user turns. Multiple tool stages can reuse the same question.
    user_sources = {}
    for w in training:
        for j, m in enumerate(w["messages"]):
            if m["role"] == "user":
                user_sources.setdefault(m["content"], f"{w['id']}/{j}")
    records = nearest(
        "complete_user_turn",
        [(v, k) for k, v in user_sources.items()],
        [(w["id"], w["family"], w["questions"][0]) for w in panel],
    )
    db = read(ROOT / "database-dialogues/originals.json")["cohorts"]
    human = read(ROOT / "human-assistance/originals.json")["splits"]
    train = [(w["tree"], w["question"]) for w in human["train"]["selected"]]
    # Use actual customer turns exposed anywhere in the enrolled CoSQL conversations, including history.
    for d in db["train"]:
        for w in d["dialogues"]:
            for j, m in enumerate(w["prior"]):
                if m["role"] in ("user", "customer"):
                    train.append(
                        (
                            f"{d['database']}/{w['source']}/{w['turn']}/prior-{j}",
                            m["content"],
                        )
                    )
            for j, req in enumerate(w["sequence"]):
                train.append(
                    (f"{d['database']}/{w['source']}/{w['turn']}/{j}", req["question"])
                )
    originals = [
        (w["id"], w["family"], w.get("original_question", w["questions"][0]))
        for w in panel
        if w["family"] in ("open", "database")
    ]
    records.extend(nearest("original_utterance", train, originals))
    result = {}
    for kind in ("complete_user_turn", "original_utterance"):
        subsets = defaultdict(list)
        for r in records:
            if r["audit"] == kind:
                subsets[r["family"]].append(r)
                subsets["all"].append(r)
        result[kind] = {
            g: {
                "n": len(rows),
                "exact": sum(r["normalized_exact_matches"] > 0 for r in rows),
                "near_at_least_0.80": sum(
                    r["nearest_fourgram_jaccard"] >= 0.8 for r in rows
                ),
                "max": max(r["nearest_fourgram_jaccard"] for r in rows),
            }
            for g, rows in subsets.items()
        }
    path = Path("paper/assistance-source-overlap.csv")
    write_csv(path, records)
    output = {
        "definition": "Unicode NFKC and casefold, word tokens, then 4-word set Jaccard. Shorter texts have one whole-text gram. Retain each request including any repeated wording. Similarity describes observed wording, not access to pretraining.",
        "unique_training_user_messages": len(user_sources),
        "comparisons": result,
        "rows": digest(path),
    }
    dest = Path("docs/measurements/integrated-assistance-source-overlap.json")
    if dest.exists():
        assert read(dest) == output
    else:
        save(dest, output)
    return result


if __name__ == "__main__":
    print({k: v["all"] for k, v in build().items()})
