"""Create independent reading transfer material with opposed hypotheses sharing a premise."""

import hashlib
import json
import random
import sys
from pathlib import Path
from urllib.request import urlopen

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from loopqwen.io import digest, read, save
from loopqwen.research.inference_data import entries
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.contrast_material import REV, URL, key, messages

DATA = Path("research/data")


def make():
    previous = read(DATA / "contrast-boolq-panel.json")
    with urlopen(URL, timeout=45) as resp:
        source = resp.read()
    assert hashlib.sha256(source).hexdigest() == previous["sha256"]
    published = json.loads(source)["data"]
    used = {r["pair"] for r in previous["rows"]}
    passages = {key(r["paragraph"]) for r in published}
    questions = {
        key(q)
        for r in published
        for q in [r["question"], *[a["perturbed_q"] for a in r["perturbed_questions"]]]
    }
    rng = random.Random(771852)
    items = []
    for r in published:
        passage_id = key(r["paragraph"])
        if passage_id in used or r["answer"] not in ("TRUE", "FALSE"):
            continue
        edits = [
            e
            for e in r["perturbed_questions"]
            if e["answer"] in ("TRUE", "FALSE") and e["answer"] != r["answer"]
        ]
        if not edits:
            continue
        used.add(passage_id)
        edit = rng.choice(edits)
        for variant, question, truth in [
            ("original", r["question"], r["answer"]),
            ("edit", edit["perturbed_q"], edit["answer"]),
        ]:
            items.append(
                dict(
                    id=passage_id + "-" + variant,
                    pair=passage_id,
                    variant=variant,
                    messages=messages(r["paragraph"], question),
                    gold="yes" if truth == "TRUE" else "no",
                )
            )
    assert len(items) == 64 and len(used) == 64
    rng.shuffle(items)
    save(
        DATA / "contrast-boolq-transfer-panel.json",
        dict(url=URL, sha256=previous["sha256"], rows=items),
    )

    original = read(DATA / "contrast-boolq-training.json")
    passages.update(r["id"] for r in original["examples"])
    path = Path(
        hf_hub_download(
            "google/boolq",
            "data/train-00000-of-00001.parquet",
            repo_type="dataset",
            revision=REV,
        )
    )
    assert digest(path) == original["sha256"]
    rows = pq.read_table(path).to_pylist()
    rng.shuffle(rows)
    counts = {True: 0, False: 0}
    natural = []
    for r in rows:
        p, q, y = r["passage"], r["question"], r["answer"]
        if (
            key(p) in passages
            or key(q) in questions
            or counts[y] >= 1024
            or len(p) > 1700
        ):
            continue
        passages.add(key(p))
        questions.add(key(q))
        counts[y] += 1
        natural.append(
            dict(
                id=key(p),
                messages=messages(p, q),
                visible="yes" if y else "no",
                thinking="",
            )
        )
    assert len(natural) == 2048
    spec = read("research/experiments/entailment-triplets-01.json")
    groups = {}
    with entries(spec, "esnli", True) as stream:
        for premise, hyp, label, _, uid, _ in stream:
            pk = key(premise)
            if (
                label not in ("entailment", "contradiction")
                or int(pk[:6], 16) % 31
                or pk in passages
                or key(hyp) in questions
            ):
                continue
            groups.setdefault(pk, {}).setdefault(label, (premise, hyp, uid))
    matched = [g for g in sorted(groups) if len(groups[g]) == 2]
    rng.shuffle(matched)
    paired = []
    for pk in matched[:512]:
        for label in ("entailment", "contradiction"):
            premise, hyp, uid = groups[pk][label]
            paired.append(
                dict(
                    id=uid,
                    pair=pk,
                    messages=messages(
                        premise, "Does the passage support the following claim? " + hyp
                    ),
                    visible="yes" if label == "entailment" else "no",
                    thinking="",
                )
            )
    assert len(paired) == 1024 and len({r["id"] for r in natural + paired}) == 3072
    save(
        DATA / "contrast-boolq-transfer-training.json",
        dict(
            boolq=original["sha256"],
            esnli=spec["files"]["esnli_train_1.csv"],
            examples=natural + paired,
        ),
    )
    print(
        len(natural),
        "BoolQ training passages,",
        len(paired) // 2,
        "opposed e-SNLI training pairs,",
        len(items) // 2,
        "independent test passages",
    )


if __name__ == "__main__":
    make()
