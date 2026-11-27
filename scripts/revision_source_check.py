"""Recover the actual Wikipedia edit and parent identities for every frozen article."""

import argparse
import gzip
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from loopqwen.io import read, save
from scripts.revision_material import PANEL

DIFFS = Path("research/diagnostics/revision-wikipedia-diffs.json.gz")
RESULT = Path("docs/measurements/revision-wikipedia-sources.json")


def api(**params):
    url = "https://en.wikipedia.org/w/api.php?" + urlencode(
        dict(action="query", format="json", **params)
    )
    req = Request(
        url,
        headers={
            "User-Agent": "LoopQwen-MLX research (https://github.com/lukifer23/LoopQwen-MLX)"
        },
    )
    with urlopen(req, timeout=45) as response:
        raw = json.load(response)
    if "error" in raw:
        raise ValueError(raw["error"])
    return raw


def recover(ids):
    q = api(prop="revisions", revids="|".join(ids), rvprop="ids|timestamp")
    docs = {}
    for page in q["query"]["pages"].values():
        for rev in page["revisions"]:
            child, parent = rev["revid"], rev["parentid"]
            url = "https://en.wikipedia.org/w/api.php?" + urlencode(
                {
                    "action": "compare",
                    "format": "json",
                    "fromrev": parent,
                    "torev": child,
                    "prop": "ids|diff|title",
                }
            )
            req = Request(
                url,
                headers={
                    "User-Agent": "LoopQwen-MLX research (https://github.com/lukifer23/LoopQwen-MLX)"
                },
            )
            with urlopen(req, timeout=45) as response:
                diff = json.load(response)["compare"]
            assert (diff["fromrevid"], diff["torevid"]) == (parent, child)
            docs[str(child)] = {
                "pageid": page["pageid"],
                "title": page["title"],
                "timestamp": rev["timestamp"],
                "parent": parent,
                "diff": diff["*"],
            }
    assert docs.keys() == set(ids)
    return dict(sorted(docs.items()))


def tokens(text):
    return re.findall(r"[^\W_]+", html.unescape(text).casefold())


def overlap(text, raw):
    # Fraction of unique excerpt trigrams present in a changed wikitext line.
    words = tokens(text)
    triples = set(zip(words, words[1:], words[2:]))
    if not triples:
        return 0
    other = tokens(raw)
    return len(triples.intersection(zip(other, other[1:], other[2:]))) / len(triples)


def main(fetch=False):
    rows = read(PANEL)
    ids = sorted({r["revision"] for r in rows})
    if fetch:
        docs = recover(ids)
        data = gzip.compress(
            json.dumps(docs, sort_keys=True, ensure_ascii=False).encode(), mtime=0
        )
        if DIFFS.exists():
            assert json.loads(gzip.decompress(DIFFS.read_bytes())) == docs
        else:
            DIFFS.write_bytes(data)
    docs = json.loads(gzip.decompress(DIFFS.read_bytes()))
    assert set(ids) == docs.keys()
    scores = []
    for key in ids:
        doc = docs[key]
        sides = []
        for cls in ("diff-deletedline", "diff-addedline"):
            cells = re.findall(
                r'<td\b[^>]*class="[^"]*\b' + cls + r'\b[^"]*"[^>]*>(.*?)</td>',
                doc["diff"],
                re.DOTALL,
            )
            sides.append(html.unescape(re.sub(r"<[^>]+>", " ", " ".join(cells))))
        pair = [r for r in rows if r["revision"] == key]
        assert len(pair) == 2 and len({r["claim"] for r in pair}) == 1
        matrix = [[overlap(r["evidence"], cell) for cell in sides] for r in pair]
        aligned = max(min(matrix[0][0], matrix[1][1]), min(matrix[1][0], matrix[0][1]))
        scores.append(
            {
                "revision": int(key),
                "parent": doc["parent"],
                "title": doc["title"],
                "timestamp": doc["timestamp"],
                "ids": [r["id"] for r in pair],
                "trigram_coverage": matrix,
                "weakest_opposite_side_coverage": aligned,
                "url": f"https://en.wikipedia.org/wiki/Special:Diff/{key}",
            }
        )
    result = {
        "edits": len(ids),
        "rows": len(rows),
        "years": dict(Counter(r["timestamp"][:4] for r in scores)),
        "source_sha256": hashlib.sha256(DIFFS.read_bytes()).hexdigest(),
        "articles": scores,
    }
    if RESULT.exists():
        assert read(RESULT) == result
    else:
        save(RESULT, result)
    print(
        len(ids),
        "distinct live Wikipedia edits; both excerpts >=50% trigram coverage in opposed changed lines:",
        sum(r["weakest_opposite_side_coverage"] >= 0.5 for r in scores),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fetch", action="store_true")
    main(p.parse_args().fetch)
