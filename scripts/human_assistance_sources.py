"""Enroll reviewed, top-ranked authentic assistance from disjoint OASST1 conversation trees."""

import argparse
import shutil
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save

REV = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
SPLITS = {
    "train": (
        "train-00000-of-00001-b42a775f407cee45.parquet",
        "bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e",
        256,
    ),
    "validation": (
        "validation-00000-of-00001-134b8fd0c89408b6.parquet",
        "24002597bb13a7edd42d92f773762f25e285f72c31a70449393d0ded1dc7b416",
        12,
    ),
}
CACHE = Path("data/research_external/human-assistance")
DEST = Path("research/sources/human-assistance")
FIELDS = (
    "message_id",
    "parent_id",
    "role",
    "lang",
    "review_result",
    "deleted",
    "synthetic",
    "rank",
    "labels",
    "text",
    "message_tree_id",
)


def eligible(row):
    if (
        row["lang"] != "en"
        or not row["review_result"]
        or row["deleted"]
        or row["synthetic"]
        or not row["labels"]
    ):
        return False
    ratings = dict(zip(row["labels"]["name"], row["labels"]["value"], strict=True))
    return ratings.get("quality", 0) >= 0.66 and all(
        ratings.get(k, 0) == 0
        for k in (
            "spam",
            "pii",
            "not_appropriate",
            "violence",
            "hate_speech",
            "sexual_content",
            "fails_task",
            "lang_mismatch",
        )
    )


def build(acquire=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    data = {}
    for split, (file, sha, cap) in SPLITS.items():
        path = CACHE / (split + ".parquet")
        if acquire and not path.is_file():
            url = f"https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/{REV}/data/{file}"
            pending = path.with_suffix(".partial")
            with urllib.request.urlopen(url, timeout=90) as r, pending.open("wb") as f:
                shutil.copyfileobj(r, f)
            assert digest(pending) == sha
            pending.replace(path)
        assert digest(path) == sha
        rows = {}
        for batch in pq.ParquetFile(path).iter_batches(
            columns=list(FIELDS), batch_size=512
        ):
            for r in batch.to_pylist():
                if eligible(r):
                    rows[r["message_id"]] = r
        chosen = []
        for r in rows.values():
            p = rows.get(r["parent_id"])
            if (
                r["role"] != "assistant"
                or r["rank"] != 0
                or not p
                or p["parent_id"] is not None
            ):
                continue
            if not (
                15 <= len(r["text"].split()) <= 200
                and 1 <= len(p["text"].split()) <= 120
            ):
                continue
            assert p["role"] == "prompter" and p["message_id"] == r["message_tree_id"]
            chosen.append(
                {
                    "tree": p["message_id"],
                    "response_id": r["message_id"],
                    "question": p["text"],
                    "answer": r["text"],
                    "parent_labels": p["labels"],
                    "response_labels": r["labels"],
                }
            )
        chosen.sort(key=lambda w: fingerprint([914731, w["tree"], w["response_id"]]))
        unique = {}
        for w in chosen:
            unique.setdefault(w["tree"], w)
        selected = list(unique.values())[:cap]
        assert len(selected) == cap
        data[split] = {
            "release_sha256": sha,
            "eligible_trees": len(unique),
            "selected": selected,
        }
    tr, ev = (data[split]["selected"] for split in SPLITS)
    assert not {x["tree"] for x in tr} & {x["tree"] for x in ev}
    assert not {x["question"].casefold().strip() for x in tr} & {
        x["question"].casefold().strip() for x in ev
    }
    export = {
        "repository": "OpenAssistant/oasst1",
        "revision": REV,
        "license": "Apache-2.0",
        "splits": data,
    }
    path = DEST / "originals.json"
    if path.is_file():
        assert read(path) == export
    else:
        save(path, export)
    print({k: (v["eligible_trees"], len(v["selected"])) for k, v in data.items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acquire", action="store_true")
    build(p.parse_args().acquire)
