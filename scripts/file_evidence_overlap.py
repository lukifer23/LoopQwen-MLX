"""Audit lexical overlap between all enrolled source evidence and actual training texts."""

import re
from pathlib import Path

from loopqwen.io import read, save


def lex(text):
    # Treat numbers as a shared symbol to detect round/date changes in a reused sentence.
    return re.findall(
        r"[^\W\d_]+|<n>", re.sub(r"\d+(?:[,.]\d+)*", "<n>", text.casefold())
    )


def shingles(words, width):
    return {tuple(words[i : i + width]) for i in range(len(words) - width + 1)}


def main():
    s = read("research/experiments/file-evidence-01.json")
    training = read(s["training"])
    held = read(s["material"])["rows"]
    articles = {a["case"]: a for a in training["source_articles"]}
    examples = training["examples"]
    refs = {"claim": {}, "evidence": {}}
    for item in articles.values():
        refs["claim"][item["case"]] = item["claim"]
    import json

    for r in examples:
        if not r.get("pair") or not r["tools"]:
            continue
        assert r["messages"][-1]["role"] == "tool"
        refs["evidence"][r["id"]] = json.loads(r["messages"][-1]["content"])["value"]
    assert len(refs["evidence"]) == 2 * len(articles)
    details = {}
    for field, targets in refs.items():
        source = {key: lex(t) for key, t in targets.items()}
        grams = {key: shingles(v, 5) for key, v in source.items()}
        eight = set().union(*(shingles(v, 8) for v in source.values()))
        rows = []
        unique = {r["cluster"] if field == "claim" else r["id"]: r[field] for r in held}
        for key, text in unique.items():
            words = lex(text)
            sh = shingles(words, 5)
            scores = {
                k: (len(sh & g) / len(sh | g) if sh and g else 0)
                for k, g in grams.items()
            }
            best = max(scores, key=scores.get)
            rows.append(
                {
                    "id": key,
                    "closest_training_id": best,
                    "normalized_exact": words in source.values(),
                    "five_word_jaccard": scores[best],
                    "shared_unique_eight_word_phrases": len(shingles(words, 8) & eight),
                }
            )
        details[field] = {
            "held_out": len(rows),
            "normalized_exact": sum(r["normalized_exact"] for r in rows),
            "any_shared_eight": sum(
                bool(r["shared_unique_eight_word_phrases"]) for r in rows
            ),
            "above_point_six": sum(r["five_word_jaccard"] >= 0.6 for r in rows),
            "items": rows,
        }
    doc = {
        "article_overlap": sorted(
            {w["page"].casefold() for w in held}
            & {r["page"].casefold() for r in articles.values()}
        ),
        "fields": details,
    }
    out = Path("docs/measurements/file-evidence-source-overlap.json")
    if out.exists():
        assert read(out) == doc
    else:
        save(out, doc)
    print({k: {x: y for x, y in v.items() if x != "items"} for k, v in details.items()})


if __name__ == "__main__":
    main()
