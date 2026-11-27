"""Check held source text and question overlap across every enrolled training article."""

import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from loopqwen.io import digest, read, save
from scripts.linked_table_material import cases

OUT = Path("docs/measurements/linked-source-overlap.json")


def words(text):
    return tuple(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def grams(tokens, n):
    return {tokens[i : i + n] for i in range(len(tokens) - n + 1)}


def run():
    panel_path = Path("research/data/linked-rollin-panel.json")
    held = {
        w["cluster"]: w
        for w in read(panel_path)["rows"]
        if w["family"] == "linked_file"
    }
    training = {}
    questions = {}
    reused_text = defaultdict(set)
    for split, w, _, pages in cases():
        if split == "train":
            questions[w["question_id"]] = words(w["question"])
            for page, text in pages.items():
                # A given page can be included with different revision text across source tables.
                training[(page, text)] = None
                reused_text[page].add(text)
    targets = {}
    for key, w in held.items():
        article = w["events"][-1]["observation"]["value"]
        targets[key] = {
            "full": words(article),
            "quote": words(w["quote"]),
            "gold": words(w["gold"]),
            "question": words(w["questions"][0].splitlines()[-1]),
        }
    tokenized = {k: words(k[1]) for k in training}
    counts = {}
    for key, v in targets.items():
        occupied = grams(v["full"], 5)
        qgrams = grams(v["question"], 3)
        best_fraction, best_source = 0.0, None
        quotes = []
        exact = []
        answers = 0
        for (page, _), tok in tokenized.items():
            # Compare normalized complete articles including punctuation and tokenization aliases.
            if tok == v["full"]:
                exact.append(page)
            if v["gold"] in grams(tok, len(v["gold"])):
                answers += 1
            if v["quote"] in grams(tok, len(v["quote"])):
                quotes.append(page)
            overlap = len(occupied & grams(tok, 5)) / len(occupied)
            if overlap > best_fraction:
                best_fraction, best_source = overlap, page
        counts[key] = {
            "same_normalized_article": sorted(set(exact)),
            "full_target_quote": sorted(set(quotes)),
            "training_passages_containing_gold": answers,
            "max_target_5gram_coverage": best_fraction,
            "nearest_training_page": best_source,
            "exact_normalized_question": any(
                v["question"] == q for q in questions.values()
            ),
            "max_question_trigram_coverage": max(
                len(qgrams & grams(q, 3)) / len(qgrams) for q in questions.values()
            ),
        }
    result = {
        "panel_sha256": digest(panel_path),
        "training_questions": len(questions),
        "training_page_ids": len(reused_text),
        "distinct_training_page_text_pairs": len(training),
        "held_questions": len(counts),
        "items": counts,
    }
    if OUT.exists():
        assert read(OUT) == result
    else:
        save(OUT, result)
    print(
        "Normalized exact held questions, articles, and entire answer-bearing quotes in training:",
        sum(r["exact_normalized_question"] for r in counts.values()),
        sum(bool(r["same_normalized_article"]) for r in counts.values()),
        sum(bool(r["full_target_quote"]) for r in counts.values()),
    )
    print(
        "Maximum target article 5-gram coverage:",
        max(r["max_target_5gram_coverage"] for r in counts.values()),
    )


if __name__ == "__main__":
    run()
