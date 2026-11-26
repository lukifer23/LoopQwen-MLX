"""Freeze disjoint BoolQ training passages and human question-edit pairs."""

import hashlib
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

from loopqwen.io import digest, save

REV = "35b264d03638db9f4ce671b711558bf7ff0f80d5"
URL = "https://raw.githubusercontent.com/allenai/contrast-sets/e2b87316ba7b30093bc9ffbc340e6550cab79f67/BoolQ/boolq_perturbed.json"


def key(text):
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def messages(passage, question):
    return [
        dict(
            role="system",
            content="Answer the user's yes/no question from the passage. Give yes or no as the final answer.",
        ),
        dict(role="user", content=f"Passage\n{passage}\n\nQuestion\n{question}"),
    ]


def main():
    import pyarrow.parquet as pq

    path = Path(
        hf_hub_download(
            "google/boolq",
            "data/train-00000-of-00001.parquet",
            repo_type="dataset",
            revision=REV,
        )
    )
    assert (
        digest(path)
        == "4f028e992c0bd4df30b9f056f4946b64f5c23028034ff0ed5ea467d8538cc623"
    )
    raw = urllib.request.urlopen(URL, timeout=30).read()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "0b4505e885d99fba40ff3d5a1f70d26839eb444cfe21d07cbf571c52f967c536"
    )
    contrast = json.loads(raw)["data"]
    exclude = {key(r["paragraph"]) for r in contrast}
    questions = {
        key(q)
        for r in contrast
        for q in [r["question"], *[e["perturbed_q"] for e in r["perturbed_questions"]]]
    }
    rng = random.Random(52914)
    rows = pq.read_table(path).to_pylist()
    rng.shuffle(rows)
    counts = Counter()
    train = []
    for r in rows:
        p, q = r["passage"], r["question"]
        h = key(p)
        if (
            h in exclude
            or key(q) in questions
            or len(p) > 1700
            or counts[r["answer"]] >= 768
        ):
            continue
        exclude.add(h)
        questions.add(key(q))
        counts[r["answer"]] += 1
        train.append(
            dict(
                id=h,
                messages=messages(p, q),
                visible="yes" if r["answer"] else "no",
                thinking="",
            )
        )
    assert len(train) == 1536
    save(
        "research/data/contrast-boolq-training.json",
        dict(
            repository="google/boolq", revision=REV, sha256=digest(path), examples=train
        ),
    )
    rng.shuffle(contrast)
    items = []
    enrolled = set()
    for r in contrast:
        p = r["paragraph"]
        if key(p) in enrolled or r["answer"] not in ("TRUE", "FALSE"):
            continue
        edits = [
            e
            for e in r["perturbed_questions"]
            if e["answer"] in ("TRUE", "FALSE") and e["answer"] != r["answer"]
        ]
        if not edits:
            continue
        edit = rng.choice(edits)
        enrolled.add(key(p))
        for variant, q, truth in [
            ("original", r["question"], r["answer"]),
            ("edit", edit["perturbed_q"], edit["answer"]),
        ]:
            items.append(
                dict(
                    id=key(p) + "-" + variant,
                    pair=key(p),
                    variant=variant,
                    messages=messages(p, q),
                    gold="yes" if truth == "TRUE" else "no",
                )
            )
        if len(items) == 64:
            break
    assert len(items) == 64
    rng.shuffle(items)
    save(
        "research/data/contrast-boolq-panel.json",
        dict(url=URL, sha256=hashlib.sha256(raw).hexdigest(), rows=items),
    )
    print(
        len(train),
        "balanced training passages and",
        len(enrolled),
        "independent contrast passages",
    )


if __name__ == "__main__":
    main()
