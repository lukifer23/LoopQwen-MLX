"""Enroll disjoint raw documents from fiction, natural history, reference and public science text."""

import hashlib
import json
import random
import re
from pathlib import Path
from urllib.request import urlopen

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from mlx_lm.utils import load_tokenizer
from raw_distillation_material import SCI

from loopqwen.data import retention3
from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot

DATA = Path("research/data")
CACHE = Path("data/research_external/selective-prose")
DOC = retention3.CPYTHON + "Doc/"
TRAIN_DOCS = (
    "library/argparse",
    "library/asyncio-task",
    "library/pathlib",
    "library/subprocess",
    "library/datetime",
    "library/logging",
    "library/typing",
    "library/concurrent.futures",
    "howto/unicode",
    "howto/descriptor",
    "library/ast",
    "library/re",
    "library/inspect",
    "library/os",
    "library/shutil",
    "library/unittest",
    "library/threading",
    "library/functools",
    "library/decimal",
    "library/contextlib",
    "library/io",
    "library/pickle",
    "library/tempfile",
)
TEST_DOCS = (
    "library/csv",
    "library/zipfile",
    "library/xml.etree.elementtree",
    "library/zoneinfo",
)
BOOK_URLS = {
    215: "https://raw.githubusercontent.com/GITenberg/The-Call-of-the-Wild_215/c1fcf9ec7af90edd09a7a2d3b6f1354b153494d0/215-0.txt",
    944: "https://raw.githubusercontent.com/GITenberg/The-Voyage-of-the-Beagle_944/a26cbdcdc1449627d1594dac877ad11396167791/944.txt",
    1342: "https://raw.githubusercontent.com/GITenberg/Pride-and-Prejudice_1342/81db45c9c48c592f0b77f01fc59e677ad0a5634e/1342-0.txt",
    2701: "https://raw.githubusercontent.com/GITenberg/Moby-Dick--Or-The-Whale_2701/bdf1948e6cd00963730971e5624e764a35f238c3/2701-0.txt",
    1228: "https://raw.githubusercontent.com/GITenberg/On-the-Origin-of-Species-By-Means-of-Natural-Selection--13-Or-the-Preservation-of-Favoured-Rac__1228/6c66c461969417cb73d63645085f0c534ea4b972/1228.txt",
    205: "https://raw.githubusercontent.com/GITenberg/Walden-and-On-The-Duty-Of-Civil-Disobedience_205/0efbef2325160440b0a7ca7e5130ef566a969985/205-0.txt",
}
TRAIN_BOOKS = (1342, 2701, 1228, 205)
TEST_BOOKS = (215, 944)


def identity(ids):
    return hashlib.sha256(
        b"".join(int(i).to_bytes(4, "little") for i in ids)
    ).hexdigest()


def download(name, url, sources):
    dest = CACHE / name
    if not dest.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(urlopen(url, timeout=60).read())
        tmp.replace(dest)
    sources[name] = {"url": url, "sha256": digest(dest)}
    return dest.read_text(encoding="utf-8-sig")


def paragraphs(text):
    return [
        " ".join(p.split()) for p in re.split(r"\n\s*\n", text) if len(p.strip()) >= 160
    ]


def book(number, sources):
    key = f"pg{number}.txt"
    url = BOOK_URLS[number]
    text = download(key, url, sources).replace("\r", "")
    boundary = list(
        re.finditer(
            r"^\*{3}\s*(?:START|END) OF .*?\*{3}.*$", text, re.IGNORECASE | re.MULTILINE
        )
    )
    if len(boundary) != 2:
        raise ValueError("Unexpected public-domain book delimiters: " + key)
    return paragraphs(text[boundary[0].end() : boundary[1].start()])


def docs(names, sources):
    return [
        (
            name,
            paragraphs(
                download(name.replace("/", "-") + ".rst", DOC + name + ".rst", sources)
            ),
        )
        for name in names
    ]


def grams(seq):
    return {tuple(seq[i : i + 24]) for i in range(len(seq) - 23)}


def main():
    tok = load_tokenizer(local_snapshot())
    sources = {}
    previous = read(DATA / "raw-distillation-training.json")["rows"]
    old = list(previous)
    for name in (
        "cache-language-train",
        "cache-language-test",
        "raw-distillation-test",
    ):
        path = DATA / (name + ".json")
        if path.exists():
            old.extend(read(path)["rows"])
    legacy = [
        json.loads(line)
        for line in (retention3.ROOT / "validation.jsonl").read_text().splitlines()
    ]
    excluded = set()
    for w in old + legacy:
        excluded.update(grams(w["ids"]))

    def select(groups, n, category):
        pools = []
        for label, texts in groups:
            texts = sorted(
                set(texts),
                key=lambda s: hashlib.sha256(("selective-7318" + s).encode()).digest(),
            )
            pools.append(iter(zip([label] * len(texts), texts, strict=True)))
        out = []
        while pools and len(out) < n:
            next_pools = []
            for it in pools:
                for label, text in it:
                    ids = tok.encode(text, add_special_tokens=False)[:129]
                    if len(ids) < 64:
                        continue
                    gs = grams(ids)
                    if gs & excluded:
                        continue
                    excluded.update(gs)
                    out.append(
                        {
                            "id": category + "-" + identity(ids)[:20],
                            "source": label,
                            "category": category,
                            "ids": ids,
                        }
                    )
                    next_pools.append(it)
                    break
                if len(out) == n:
                    break
            pools = next_pools
        if len(out) != n:
            raise ValueError((category, len(out), n))
        return out

    # Enroll all fresh evaluation material before selecting any new fitting examples.
    new = []
    for number, group in zip(
        TEST_BOOKS, ("new_fiction", "new_natural_history"), strict=True
    ):
        new.extend(select([(f"pg{number}", book(number, sources))], 32, group))
    new.extend(select(docs(TEST_DOCS, sources), 32, "new_documentation"))
    science = Path(
        hf_hub_download(
            "allenai/sciq",
            "data/test-00000-of-00001.parquet",
            revision=SCI,
            repo_type="dataset",
        )
    )
    sources["sciq-test"] = {"revision": SCI, "sha256": digest(science)}
    vals = pq.read_table(science, columns=["support"]).column(0).to_pylist()
    new.extend(select([("SciQ test", [v for v in vals if v])], 32, "new_science"))
    added = []
    for number in TRAIN_BOOKS:
        added += select([(f"pg{number}", book(number, sources))], 128, "book")
    added += select(docs(TRAIN_DOCS, sources), 512, "documentation")
    rng = random.Random(719916)
    for r in added:
        r["positions"] = sorted(rng.sample(range(8, len(r["ids"]) - 1), 32))
    assert len(previous) == len(added) == 1024 and len(new) == 128
    # The sealed studies used the final iterator label on the multi-manual groups.
    # Reproduce their immutable metadata. The publisher audit records the actual work
    # for every token-identical row; new consumers use those corrected identities.
    for rows, category, archived_label in (
        (new, "new_documentation", TEST_DOCS[-1]),
        (added, "documentation", TRAIN_DOCS[-1]),
    ):
        for row in rows:
            if row["category"] == category:
                row["source"] = archived_label
    save(
        DATA / "selective-raw-training.json",
        {
            "previous": digest(DATA / "raw-distillation-training.json"),
            "sources": sources,
            "rows": previous + added,
        },
    )
    save(DATA / "selective-raw-panel.json", {"sources": sources, "rows": new})
    print(len(previous) + len(added), len(new), flush=True)


if __name__ == "__main__":
    main()
