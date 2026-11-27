"""Enroll original consecutive human questions and answers, excluding earlier project sources."""

import re
from collections import Counter
from pathlib import Path

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from scripts.passage_copy_material import ROOT, SOURCES, normalized

DEST = Path("research/sources/followup-reading")


def words(text):
    return normalized(text).split()


def grams(text):
    seq = words(text)
    return {tuple(seq[j : j + 5]) for j in range(len(seq) - 4)}


def past_sources():
    titles = set()
    passages = set()
    for split in ("dev", "train"):
        for r in read(f"research/data/passage-copy-{split}.json")["rows"]:
            titles.add(normalized(r["title"]))
            passages.add(normalized(r["passage"]))
    for name in ("natural-context-01", "prose-reading-01"):
        titles.update(
            normalized(t) for t in read(f"research/experiments/{name}.json")["titles"]
        )
    titles.update(
        normalized(w["title"])
        for w in read("research/data/document-evidence-training.json")["episodes"]
    )
    titles.update(
        normalized(w["title"])
        for w in read("research/data/document-evidence-panel.json")["documents"]
    )
    titles.update(
        normalized(r["source"])
        for r in read("research/data/conditional-language-02.json")
        if r["family"] == "reading"
    )
    return titles, passages


def item(article, paragraph):
    source = paragraph["context"].removesuffix(" CANNOTANSWER")
    if not source or source == paragraph["context"]:
        return None
    turns = []
    for q in paragraph["qas"][:2]:
        labels = q["answers"]
        teacher = q.get("orig_answer", labels[0])
        # Keep annotator disagreement and abstentions. Skip malformed character anchors.
        for a in [teacher, *labels]:
            pos, text = a["answer_start"], a["text"]
            if paragraph["context"][pos : pos + len(text)] != text:
                return None
        turns.append(
            {
                "id": q["id"],
                "question": q["question"],
                "teacher": teacher,
                "annotations": labels,
            }
        )
    if len(turns) != 2:
        return None
    return {
        "id": paragraph["id"],
        "title": article["title"],
        "section": article["section_title"],
        "background": article["background"],
        "passage": source,
        "turns": turns,
    }


def enroll():
    past_titles, past_passages = past_sources()
    used_titles = set(past_titles)
    used_text = set(past_passages)
    seen_grams = []
    cohort = {}
    worst = 0
    for split, size in (("dev", 24), ("train", 128)):
        file, url, sha = SOURCES["QuAC", split]
        assert digest(ROOT / file) == sha
        articles = read(ROOT / file)["data"]
        pool = sorted(
            (
                (p["id"], article, p)
                for article in articles
                for p in article["paragraphs"]
            ),
            key=lambda x: fingerprint([740033, split, x[0]]),
        )
        selected = []
        for _, article, paragraph in pool:
            title = normalized(article["title"])
            if title in used_titles:
                continue
            v = item(article, paragraph)
            if v is None or normalized(v["passage"]) in used_text:
                continue
            g = grams(v["passage"])
            if not g:
                continue
            fraction = max(
                (len(g & h) / min(len(g), len(h)) for h in seen_grams), default=0
            )
            if fraction >= 0.8:
                continue
            worst = max(worst, fraction)
            selected.append(v)
            seen_grams.append(g)
            used_titles.add(title)
            used_text.add(normalized(v["passage"]))
            if len(selected) == size:
                break
        assert len(selected) == size
        cohort[split] = {"source": url, "sha256": sha, "rows": selected}
    counts = {}
    for split, catalog in cohort.items():
        rows = catalog["rows"]
        allq = [q for r in rows for q in r["turns"]]
        for r in rows:
            assert re.sub(r"_q#\d+$", "", r["turns"][0]["id"]) == r["id"]
            assert [q["id"] for q in r["turns"]] == [
                r["id"] + f"_q#{i}" for i in range(2)
            ]
        counts[split] = {
            "dialogues": len(rows),
            "human_questions": len(allq),
            "span_annotations": sum(len(q["annotations"]) for q in allq),
            "null_majorities": sum(
                sum(a["text"] == "CANNOTANSWER" for a in q["annotations"]) * 2
                >= len(q["annotations"])
                for q in allq
            ),
            "distinct_reference_counts": dict(
                sorted(
                    Counter(
                        str(len({a["text"] for a in q["annotations"]})) for q in allq
                    ).items()
                )
            ),
        }
    for file, data in (
        ("originals", cohort),
        (
            "audit",
            {
                "seed": 740033,
                "counts": counts,
                "historical_excluded_titles": len(past_titles),
                "historical_excluded_passages": len(past_passages),
                "maximum_within_or_across_new_cohorts_shorter_passage_5gram_coverage": worst,
            },
        ),
    ):
        path = DEST / (file + ".json")
        if path.exists():
            assert read(path) == data
        else:
            save(path, data)
    print(counts)


if __name__ == "__main__":
    enroll()
