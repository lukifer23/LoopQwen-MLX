"""Enroll contrast pairs with one shared text and opposite human judgments."""

import gzip
import hashlib
import random
import re
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from mlx_lm.utils import load_tokenizer

from loopqwen.identity import fingerprint
from loopqwen.io import digest, save
from loopqwen.meaning_alignment import request
from loopqwen.model_loader import local_snapshot

DATA = Path("research/data")
CACHE = Path("data/research_external/meaning")
REV = "161ece9501cf0a11f3e48bd356eaa82de46d6a09"
UBUNTU = "https://raw.githubusercontent.com/taolei87/askubuntu/4d0a9169ef0bfa4c93ff1aa9f25bc426a7fa559e/"
PROMPTS = {
    "Wikipedia": "Do the two texts convey the same meaning? Answer Yes or No.",
    "AskUbuntu": "Do the two Ubuntu questions describe a similar problem, such that a solution to one would be relevant to the other? Answer Yes or No.",
}


def normal(t):
    return " ".join(re.findall(r"\w+", t.casefold()))


def wiki(split, sources):
    key = f"labeled_final/{split}-00000-of-00001.parquet"
    f = Path(
        hf_hub_download(
            "google-research-datasets/paws", key, repo_type="dataset", revision=REV
        )
    )
    sources[key] = {"revision": REV, "sha256": digest(f)}
    by = defaultdict(lambda: [[], []])
    for row in pq.read_table(f).to_pylist():
        if normal(row["sentence1"]) != normal(row["sentence2"]):
            by[row["sentence1"]][row["label"]].append(row)
    return by


def pick_wiki(by, n, rng, excluded, split):
    anchors = sorted(by)
    rng.shuffle(anchors)
    out = []
    for left in anchors:
        if normal(left) in excluded or not all(by[left]):
            continue
        rs = []
        for label, alternatives in enumerate(by[left]):
            candidates = [
                r for r in alternatives if normal(r["sentence2"]) not in excluded
            ]
            if not candidates:
                break
            r = rng.choice(candidates)
            rs.append(
                dict(
                    id=f"paws-{split}-{r['id']}",
                    domain="Wikipedia",
                    left=left,
                    right=r["sentence2"],
                    label=label,
                )
            )
        if len(rs) < 2 or normal(rs[0]["right"]) == normal(rs[1]["right"]):
            continue
        for r in rs:
            r["pair"] = f"wiki-{rs[0]['id']}-{rs[1]['id']}"
            excluded.update([normal(r["left"]), normal(r["right"])])
        out.extend(rs)
        if len(out) == 2 * n:
            return out
    raise ValueError("Too few disjoint PAWS anchors")


def ubuntu(n, rng, sources):
    files = {}
    for key in ("test.txt", "text_tokenized.txt.gz"):
        f = CACHE / key
        if not f.exists():
            CACHE.mkdir(parents=True, exist_ok=True)
            f.write_bytes(urlopen(UBUNTU + key, timeout=60).read())
        sources[key] = {"url": UBUNTU + key, "sha256": digest(f)}
        files[key] = f
    groups = list(files["test.txt"].read_text().splitlines())
    rng.shuffle(groups)
    seen = set()
    rows = []
    for line in groups:
        anchor, good, ranked, _ = line.split("\t")
        good = set(good.split())
        pool = ranked.split()
        if not good or anchor in seen:
            continue
        # Both are actual annotator judgments on the publisher's BM25 candidates.
        pos = next(
            (v for v in pool if v in good and v not in seen and v != anchor), None
        )
        neg = next(
            (v for v in pool if v not in good and v not in seen and v != anchor), None
        )
        if not pos or not neg:
            continue
        seen.update([anchor, pos, neg])
        rows.extend(
            dict(
                id=f"ubuntu-{anchor}-{item}",
                pair=f"ubuntu-{anchor}",
                domain="AskUbuntu",
                posts=[anchor, item],
                label=label,
            )
            for label, item in enumerate([neg, pos])
        )
        if len(rows) == 2 * n:
            break
    assert len(rows) == 2 * n
    txt = {}
    with gzip.open(files["text_tokenized.txt.gz"], "rt") as stream:
        for line in stream:
            key, title, body = line.rstrip("\n").split("\t", 2)
            if key in seen:
                txt[key] = title + "\n" + body
    for r in rows:
        r["left"], r["right"] = [txt[i] for i in r["posts"]]
    return rows


def enrich(rows, tok, freeze):
    for r in rows:
        _, ids, spans = request(tok, r["left"], r["right"], PROMPTS[r["domain"]])
        r.update(sha256=fingerprint(ids), spans=spans)
        if freeze:
            r["prompt_ids"] = ids


def main():
    sources = {}
    rng = random.Random(155131)
    excluded = set()
    panel = pick_wiki(wiki("validation", sources), 24, rng, excluded, "validation")
    panel += ubuntu(16, rng, sources)
    heldout = set(excluded)
    training = pick_wiki(wiki("train", sources), 1536, rng, excluded, "train")
    assert not {normal(r[k]) for r in training for k in ("left", "right")} & heldout
    rng.shuffle(panel)
    tok = load_tokenizer(local_snapshot())
    for rows, size, file in [(training, 3072, "training"), (panel, 80, "panel")]:
        assert len(rows) == size
        enrich(rows, tok, file == "panel")
        save(
            DATA / f"meaning-{file}.json",
            dict(
                instructions=PROMPTS,
                sources=sources,
                rows=rows,
                attribution="PAWS-Wiki from Google LLC, Zhang, Baldridge and He (2019), with the publisher's permissive dataset notice at https://github.com/google-research-datasets/paws/blob/master/LICENSE. AskUbuntu contributions from the 2014 corpus via Lei et al. (2016), original post and author at https://askubuntu.com/q/POST_ID for each listed ID. Original contributions are subject to their Stack Exchange Creative Commons Attribution-ShareAlike license, https://askubuntu.com/help/licensing. The authors' preprocessing lowercased and tokenized posts and limited bodies to 100 words; this release pairs them under the original relevance judgments.",
            ),
        )
    print(len(training), len(panel), len(heldout))


if __name__ == "__main__":
    main()
