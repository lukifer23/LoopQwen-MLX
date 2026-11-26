"""Freeze disjoint works for recurrent next-token learning and transfer on real text."""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from zipfile import ZipFile

from mlx_lm.utils import load_tokenizer
from selective_raw_material import TEST_DOCS, TRAIN_DOCS, grams, identity

from loopqwen.data import retention3
from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot

DATA = Path("research/data")
CACHE = Path("data/research_external/source-balanced")
PYTHON = "4a5632fbf9bf59477c540e3f53fa7cdbeea3e3f5"
SEED = 629378


def cached(key, url):
    path = CACHE / key
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url, timeout=90) as source:
            blob = source.read()
        tmp = path.with_suffix(".part")
        tmp.write_bytes(blob)
        tmp.replace(path)
    return path


def ordered(items, salt):
    return sorted(
        items, key=lambda x: hashlib.sha256((salt + str(x)).encode()).digest()
    )


def library(split, source_files):
    key = split + "-objects.json"
    params = urlencode(
        {
            "prefix": split + "/",
            "maxResults": 1000,
            "fields": "items(name,size,generation)",
        }
    )
    index = cached(
        key,
        "https://storage.googleapis.com/storage/v1/b/deepmind-gutenberg/o?" + params,
    )
    source_files[key] = digest(index)
    objs = {
        o["name"]: o
        for o in read(index)["items"]
        if 90000 <= int(o["size"]) <= 1300000 and o["name"].endswith(".txt")
    }
    for name in ordered(objs, str(SEED)):
        obj = objs[name]
        url = (
            "https://storage.googleapis.com/deepmind-gutenberg/"
            + name
            + "?generation="
            + obj["generation"]
        )
        path = cached(name, url)
        yield (
            "PG19/" + name,
            str(obj["generation"]),
            path.read_text(encoding="utf-8-sig"),
            digest(path),
        )


def existing():
    rows = []
    for name in (
        "future-token-panel",
        "raw-distillation-test",
        "selective-raw-panel",
        "selective-raw-training",
        "cache-language-test",
        "cache-language-train",
    ):
        rows.extend(read(DATA / (name + ".json"))["rows"])
    rows.extend(
        json.loads(x)
        for x in (retention3.ROOT / "validation.jsonl").read_text().splitlines()
    )
    return rows


def main():
    tok = load_tokenizer(local_snapshot())
    blocked = set()
    for w in existing():
        ids = w.get("ids")
        if ids is None:
            ids = tok.encode(w["text"], add_special_tokens=False)
        blocked.update(grams(ids))
    source_files = {}
    archives = {}
    for split in ("train", "dev"):
        url = (
            "https://rajpurkar.github.io/SQuAD-explorer/dataset/" + split + "-v1.1.json"
        )
        path = cached(split + "-squad.json", url)
        archives[split] = read(path)["data"]
        source_files[url] = digest(path)
    url = "https://codeload.github.com/python/cpython/zip/" + PYTHON
    path = cached("cpython.zip", url)
    source_files[url] = digest(path)
    prior_docs = set(TRAIN_DOCS) | set(TEST_DOCS)
    prior_wiki = {
        r["source"].replace("_", " ").casefold()
        for r in read(DATA / "future-token-panel.json")["rows"]
    }

    def articles(split):
        for item in ordered([w["title"] for w in archives[split]], str(SEED)):
            if item.replace("_", " ").casefold() in prior_wiki:
                continue
            data = next(w for w in archives[split] if w["title"] == item)
            text = "\n\n".join(p["context"] for p in data["paragraphs"])
            yield (
                "SQuAD/" + item,
                split + "-v1.1",
                text,
                hashlib.sha256(text.encode()).hexdigest(),
            )

    works = []
    training = []
    test = []
    rng = random.Random(SEED)
    with ZipFile(path) as cpython:
        names = ordered(
            [
                n
                for n in cpython.namelist()
                if "/Doc/library/" in n
                and n.endswith(".rst")
                and "library/" + Path(n).stem not in prior_docs
            ],
            str(SEED),
        )

        def manuals(paths):
            for name in paths:
                payload = cpython.read(name)
                yield (
                    "CPython/" + name.split("/Doc/")[1],
                    PYTHON,
                    payload.decode(),
                    hashlib.sha256(payload).hexdigest(),
                )

        enrolled = set()

        def add(group, split, candidates, number, per):
            out = training if split == "train" else test
            found = 0
            for source, rev, text, sha in candidates:
                if source in enrolled:
                    continue
                # Select contiguous, nonoverlapping chunks within a work. A bounded
                # sample of the interior avoids title pages and terminal boilerplate.
                ids = tok.encode(text.strip(), add_special_tokens=False)
                start, stop = int(0.06 * len(ids)), int(0.94 * len(ids))
                offsets = list(range(start, stop - 128, 129))
                if len(offsets) < per:
                    continue
                rng.shuffle(offsets)
                work_rows = []
                work_grams = set()
                for off in offsets:
                    window = ids[off : off + 129]
                    gram = grams(window)
                    if not gram.isdisjoint(blocked) or not gram.isdisjoint(work_grams):
                        continue
                    entry = {
                        "id": identity(window),
                        "source": source,
                        "category": group,
                        "offset": off,
                        "ids": window,
                    }
                    if split == "train":
                        entry["positions"] = sorted(rng.sample(range(128), 32))
                    work_rows.append(entry)
                    work_grams.update(gram)
                    if len(work_rows) == per:
                        break
                if len(work_rows) != per:
                    continue
                enrolled.add(source)
                blocked.update(work_grams)
                out.extend(work_rows)
                works.append(
                    {
                        "source": source,
                        "revision": rev,
                        "sha256": sha,
                        "split": split,
                        "category": group,
                        "whole_tokens": len(ids),
                        "windows": per,
                    }
                )
                found += 1
                if found == number:
                    break
            if found != number:
                raise ValueError(f"{group} {split} supplied {found}/{number} works")

        # Enroll validation first. Every fresh work has exactly four passages.
        add("books", "test", library("test", source_files), 8, 4)
        add("manuals", "test", manuals(names[:52]), 8, 4)
        add("encyclopedia", "test", articles("dev"), 8, 4)
        add("books", "train", library("train", source_files), 64, 16)
        add("manuals", "train", manuals(names[52:]), 128, 8)
        add("encyclopedia", "train", articles("train"), 128, 8)
    assert Counter(w["category"] for w in training) == {
        "books": 1024,
        "manuals": 1024,
        "encyclopedia": 1024,
    }
    save(
        DATA / "source-balanced-training.json",
        {"sources": source_files, "works": works, "rows": training},
    )
    save(
        DATA / "source-balanced-panel.json",
        {"works": [w for w in works if w["split"] == "test"], "rows": test},
    )
    print(len(training), len(test), len(works), flush=True)


if __name__ == "__main__":
    main()
