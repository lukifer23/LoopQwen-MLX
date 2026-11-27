"""Enroll genuinely different published questions about the same situation, grouped by background."""

import argparse
import json
import re
import tarfile
from pathlib import Path
from urllib.request import urlopen

from loopqwen.evaluation.extractive import normalized
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save

URL = "https://ropes-dataset.s3-us-west-2.amazonaws.com/train_and_dev/ropes-train-dev-v1.0.tar.gz"
SHA = "ba26329832f84d8c2660de1120a1d7ff086cdf8580eabeebd721c092020763e2"
ARCHIVE = Path("data/research_external/ropes-train-dev-v1.0.tar.gz")
DEST = Path("research/sources/qualitative-effects")


def words(text):
    return re.findall(r"\w+", text.casefold())


def grams(text):
    ws = words(text)
    return set(zip(*(ws[i:] for i in range(5))))


def choose(paragraph):
    pairs = []
    for a, b in zip(paragraph["qas"], paragraph["qas"][1:]):
        left = {normalized(v["text"]) for v in a["answers"]}
        right = {normalized(v["text"]) for v in b["answers"]}
        x, y = [set(words(q["question"])) for q in (a, b)]
        if not left or not right or left & right or x == y:
            continue
        similarity = len(x & y) / len(x | y)
        if similarity >= 0.35:
            pairs.append((similarity, fingerprint([771910, a["id"], b["id"]]), a, b))
    if not pairs:
        return None
    _, _, a, b = max(pairs, key=lambda z: z[:2])
    return {
        "cluster": fingerprint([paragraph["background"], paragraph["situation"]])[:20],
        "background": paragraph["background"],
        "situation": paragraph["situation"],
        "questions": [a, b],
    }


def enroll(acquire=False):
    if acquire and not ARCHIVE.exists():
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        ARCHIVE.write_bytes(urlopen(URL, timeout=45).read())
    assert digest(ARCHIVE) == SHA
    with tarfile.open(ARCHIVE) as bundle:
        groups = {}
        for split in ("dev", "train"):
            src = json.load(
                bundle.extractfile("ropes-train-dev-v1.0/" + split + "-v1.0.json")
            )
            groups[split] = [p for g in src["data"] for p in g["paragraphs"]]
    original = [{normalized(w["background"]) for w in v} for v in groups.values()]
    held_phrases = []
    seen_background = set()
    seen_situation = set()
    seen_question = set()
    selected = {}
    max_overlap = 0
    for split, need in (("dev", 24), ("train", 192)):
        rows = []
        candidates = [r for p in groups[split] if (r := choose(p))]
        candidates.sort(key=lambda r: fingerprint([771911, r["cluster"]]))
        for w in candidates:
            bg, st = normalized(w["background"]), normalized(w["situation"])
            qs = {normalized(q["question"]) for q in w["questions"]}
            if bg in seen_background or st in seen_situation or qs & seen_question:
                continue
            sections = [grams(w[k]) for k in ("background", "situation")]
            overlap = max(
                (
                    len(a & b) / min(len(a), len(b))
                    for a in sections
                    for b in held_phrases
                    if a and b
                ),
                default=0.0,
            )
            if overlap > 0.6:
                continue
            if split == "dev":
                held_phrases.extend(sections)
            else:
                max_overlap = max(max_overlap, overlap)
            rows.append(w)
            seen_background.add(bg)
            seen_situation.add(st)
            seen_question.update(qs)
            if len(rows) == need:
                break
        assert len(rows) == need
        selected[split] = rows
    outputs = {
        "source": URL,
        "sha256": SHA,
        "original_backgrounds_shared": len(original[0] & original[1]),
        "train_max_held_five_word_overlap": max_overlap,
        **selected,
    }
    path = DEST / "originals.json"
    if path.exists():
        assert read(path) == outputs
    else:
        save(path, outputs)
    print(
        "Held backgrounds",
        len(selected["dev"]),
        "training",
        len(selected["train"]),
        "original overlap",
        outputs["original_backgrounds_shared"],
        "selected maximum",
        max_overlap,
    )
    return outputs


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acquire", action="store_true")
    enroll(p.parse_args().acquire)
