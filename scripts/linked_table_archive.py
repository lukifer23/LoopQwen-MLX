"""Reconstruct the linked human source cohort from frozen original questions, tables and passages."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from loopqwen.io import digest, read, save
from scripts.linked_table_sources import CACHE, DEST, TRACED, TREES, verify_case

Z = DEST / "original-records.jsonl.gz"
OUT = Path("docs/measurements/linked-table-source-audit.json")


def freeze(enrollment):
    lines = []
    for split, rows in enrollment["items"].items():
        path = CACHE / (split + ".traced.json")
        assert digest(path) == TRACED[split]
        original = {r["question_id"]: r for r in read(path)}
        for w in rows:
            blobs = {
                k: (CACHE / k / (w["table"] + ".json")).read_bytes().decode()
                for k in TREES
            }
            lines.append(
                json.dumps(
                    {"split": split, "annotation": original[w["question_id"]], **blobs},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
    raw = ("\n".join(lines) + "\n").encode()
    if Z.exists():
        assert gzip.decompress(Z.read_bytes()) == raw
        return
    with (
        Z.open("wb") as f,
        gzip.GzipFile(filename="", fileobj=f, mode="wb", mtime=0) as z,
    ):
        z.write(raw)


def audit(enrollment):
    rows = [json.loads(v) for v in gzip.decompress(Z.read_bytes()).splitlines()]
    docs = {s: set() for s in enrollment["items"]}
    answers = {s: set() for s in docs}
    total = {
        s: {"questions": 0, "rows": 0, "linked_articles": 0, "source_bytes": 0}
        for s in docs
    }
    original = {
        (s, w["question_id"]): w
        for s, items in enrollment["items"].items()
        for w in items
    }
    seen = set()
    for r in rows:
        q = r["annotation"]
        split = r["split"]
        ident = split, q["question_id"]
        assert ident not in seen
        seen.add(ident)
        raw = {k: r[k].encode() for k in TREES}
        src = {k: json.loads(buf) for k, buf in raw.items()}
        record = verify_case(
            q, src, {k: hashlib.sha256(buf).hexdigest() for k, buf in raw.items()}
        )
        assert record == original[ident]
        docs[split].update(src["request_tok"])
        answers[split].add(record["link"])
        t = total[split]
        t["questions"] += 1
        t["rows"] += record["rows"]
        t["linked_articles"] += len(src["request_tok"])
        t["source_bytes"] += sum(map(len, raw.values()))
    assert seen == set(original) and (len(answers["dev"]), len(answers["train"])) == (
        24,
        128,
    )
    assert not (answers["dev"] & docs["train"] or answers["train"] & docs["dev"])
    report = {
        "archive": digest(Z),
        "enrollment": digest(DEST / "enrollment.json"),
        "original_question_sources": TRACED,
        "cohorts": total,
        "shared_distractor_articles": len(docs["dev"] & docs["train"]),
        "held_answer_articles_anywhere_in_training": 0,
        "training_answer_articles_anywhere_in_held": 0,
        "source": "HybridQA publisher train/dev labels and independently checked original WikiTables links",
    }
    if OUT.exists():
        assert report == read(OUT)
    else:
        save(OUT, report)
    print(
        "Verified",
        len(rows),
        "complete original question/table/passage bundles.",
        total,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    e = read(DEST / "enrollment.json")
    if args.freeze:
        freeze(e)
    audit(e)
