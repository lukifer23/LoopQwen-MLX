"""Replace complete overlapping training sources while preserving all held questions and files."""

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from loopqwen.io import digest, read, save
from scripts.linked_source_overlap import grams, words
from scripts.linked_table_sources import (
    CACHE,
    DEST,
    TRACED,
    TREES,
    blobs,
    tree_members,
    verify_case,
)

PATCH = DEST / "text-disjoint-replacements.json.gz"
RECEIPT = Path("docs/measurements/linked-text-repair.json")


def record(text):
    tokens = words(text)
    return " " + " ".join(tokens) + " ", grams(tokens, 5)


class HeldArticles:
    def __init__(self, sources):
        from scripts.linked_rollin_material import quotation

        self.ids = set()
        self.targets = {}
        self.passages = {}
        self.questions = set()
        for split, w, _, pages in sources:
            if split != "dev":
                continue
            self.questions.add(words(w["question"]))
            self.ids.update(pages)
            for k, v in pages.items():
                self.passages.setdefault((k, v), record(v))
            article = pages[w["link"]]
            quote = " " + " ".join(words(quotation(article, w["answer"]))) + " "
            self.targets[w["link"]] = (*record(article), quote)

    def overlap(self, w, pages):
        from scripts.linked_rollin_material import quotation

        why = set()
        if words(w["question"]) in self.questions:
            why.add("identical question")
        if w["link"] in self.ids or self.targets.keys() & pages.keys():
            why.add("page identifier")
        train = {k: record(v) for k, v in pages.items()}

        def check(target, others):
            body, shingles, quote = target
            for text, pieces in others:
                if body == text:
                    why.add("identical normalized article")
                if quote in text:
                    why.add("complete answer quotation")
                if shingles and 5 * len(shingles & pieces) >= 4 * len(shingles):
                    why.add("at least 80 percent target 5-gram coverage")

        for target in self.targets.values():
            check(target, train.values())
        q = " " + " ".join(words(quotation(pages[w["link"]], w["answer"]))) + " "
        check((*train[w["link"]], q), self.passages.values())
        return sorted(why)


def read_patch():
    return json.loads(gzip.decompress(PATCH.read_bytes()))


def corrected(original):
    patch = read_patch()
    replacements = {r["replaces"]: r for r in patch}
    seen = set()
    for split, w, t, pages in original:
        k = w["question_id"]
        if split == "train" and k in replacements:
            r = replacements[k]
            src = {key: json.loads(r[key]) for key in TREES}
            row = verify_case(
                r["annotation"],
                src,
                {key: hashlib.sha256(r[key].encode()).hexdigest() for key in TREES},
            )
            assert row == r["source"]
            seen.add(k)
            yield split, row, src["tables_tok"], src["request_tok"]
        else:
            yield split, w, t, pages
    assert seen == replacements.keys()


def freeze(acquire):
    from scripts.linked_table_material import cases

    sources = list(cases())
    held = HeldArticles(sources)
    existing = {w["table"] for _, w, _, _ in sources}
    forbidden = {
        str(v).casefold()
        for _, w, _, _ in sources
        for v in (w["table_url"], w["table_title"], w["link"], w["question"])
    }
    removed = [
        (w["question_id"], why)
        for split, w, _, pages in sources
        if split == "train" and (why := held.overlap(w, pages))
    ]
    trees = {k: tree_members(k, acquire) for k in TREES}
    full = CACHE / "train.traced.json"
    assert digest(full) == TRACED["train"]
    groups = defaultdict(list)
    for w in read(full):
        nodes = w["answer-node"]
        if (
            len(nodes) == 1
            and nodes[0][-1] == "passage"
            and w["table_id"] + ".json" in trees["tables_tok"]
        ):
            groups[w["table_id"]].append(w)

    def rank(k):
        return hashlib.sha256(("linked-source-61271/" + k).encode()).digest()

    last = max(rank(w["table"]) for split, w, _, _ in sources if split == "train")
    patch = []
    for table in sorted(groups, key=rank):
        if len(patch) == len(removed):
            break
        if table in existing or rank(table) <= last:
            continue
        src = blobs(table, trees, acquire)
        raw = {k: (CACHE / k / (table + ".json")).read_bytes() for k in TREES}
        for w in sorted(groups[table], key=lambda w: rank(w["question_id"])):
            row = verify_case(
                w, src, {key: hashlib.sha256(v).hexdigest() for key, v in raw.items()}
            )
            if row is None:
                continue
            ids = {
                str(row[k]).casefold()
                for k in ("table_url", "table_title", "link", "question")
            }
            if ids & forbidden or held.overlap(row, src["request_tok"]):
                continue
            old, why = removed[len(patch)]
            patch.append(
                {
                    "replaces": old,
                    "reason": why,
                    "source": row,
                    "annotation": w,
                    **{k: v.decode() for k, v in raw.items()},
                }
            )
            forbidden.update(ids)
            break
    assert len(patch) == len(removed) > 0
    if PATCH.exists():
        assert read_patch() == patch
    else:
        with (
            PATCH.open("wb") as file,
            gzip.GzipFile(fileobj=file, filename="", mode="wb", mtime=0) as z,
        ):
            z.write(
                json.dumps(patch, ensure_ascii=False, separators=(",", ":")).encode()
            )


def audit():
    from scripts.linked_table_material import cases

    old = list(cases())
    held = HeldArticles(old)
    new = list(corrected(old))
    dev = [w for w in new if w[0] == "dev"]
    assert dev == [w for w in old if w[0] == "dev"] and len(dev) == 24
    train = [(w, p) for s, w, _, p in new if s == "train"]
    assert len(train) == len({w["question_id"] for w, _ in train}) == 128
    for w, pages in train:
        assert not held.overlap(w, pages), w["question_id"]
    result = {
        "replacement_source": digest(PATCH),
        "original_source": digest(DEST / "original-records.jsonl.gz"),
        "changed_training_questions": [
            {
                "old": r["replaces"],
                "new": r["source"]["question_id"],
                "reason": r["reason"],
            }
            for r in read_patch()
        ],
        "held_questions": len(dev),
        "identical_held_records": True,
        "training_questions": len(train),
        "checks": "Both directions: no target page ID, complete normalized target article, complete answer-bearing quotation, or passage sharing >=80% of the target’s distinct normalized word 5-grams in the other split. No identical normalized train/held question.",
    }
    if RECEIPT.exists():
        assert read(RECEIPT) == result
    else:
        save(RECEIPT, result)
    print(
        "Unchanged held questions:",
        len(dev),
        "complete training sources replaced:",
        len(read_patch()),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--acquire", action="store_true")
    a = p.parse_args()
    if a.freeze:
        freeze(a.acquire)
    audit()
