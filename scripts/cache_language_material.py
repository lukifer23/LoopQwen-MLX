"""Freeze WikiText-2 article excerpts from disjoint publisher splits for raw next-token prediction."""

import hashlib
import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from mlx_lm.utils import load_tokenizer

from loopqwen.io import digest, save
from loopqwen.model_loader import local_snapshot

REV = "b08601e04326c79dfdd32d625aee71d232d685c3"
REPO = "Salesforce/wikitext"


def excerpts(split, count, used, tok):
    filename = "wikitext-2-raw-v1/" + split + "-00000-of-00001.parquet"
    p = Path(hf_hub_download(REPO, filename, repo_type="dataset", revision=REV))
    pages = {}
    title = None
    for line in pq.read_table(p, columns=["text"]).column(0).to_pylist():
        h = re.fullmatch(r"\s*= ([^=]+?) =\s*", line)
        if h:
            title = h[1].strip()
            pages.setdefault(title, [])
        elif title and line.strip():
            pages[title].append(line)
    rows = []
    for title in sorted(
        pages, key=lambda t: hashlib.sha256(("memory-lm-7481:" + t).encode()).digest()
    ):
        key = title.casefold()
        if key in used:
            continue
        text = "".join(pages[title])
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) < 513:
            continue
        offset = int.from_bytes(
            hashlib.sha256(title.encode()).digest()[:4], "little"
        ) % (len(ids) - 256)
        kept = ids[offset : offset + 257]
        if len(kept) != 257:
            raise ValueError("Excerpt boundary")
        used.add(key)
        rows.append(dict(id=title, category="wikitext", offset=offset, ids=kept))
        if len(rows) == count:
            break
    assert len(rows) == count, (split, len(pages), len(rows))
    return dict(repo=REPO, rev=REV, file=filename, source_sha256=digest(p), rows=rows)


if __name__ == "__main__":
    t = load_tokenizer(local_snapshot())
    seen = set()
    for split, count in [("train", 96), ("test", 40)]:
        data = excerpts(split, count, seen, t)
        save("research/data/cache-language-" + split + ".json", data)
        print(split, len(data["rows"]), data["source_sha256"], flush=True)
