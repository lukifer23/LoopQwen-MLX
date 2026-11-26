"""Frozen question and fact splits from the public QASC science benchmark."""

import hashlib
import json
import re
import tarfile
from pathlib import Path
from urllib.request import urlopen

from mlx_lm.utils import load_tokenizer

from ..identity import fingerprint
from ..io import digest
from ..model_loader import local_snapshot

URL = "https://ai2-public-datasets.s3.amazonaws.com/qasc/qasc_dataset.tar.gz"
SHA = "a7b3f2244f768974c609fd621346c931a72715609f171cb5544fc1da2a2ad55c"
LETTERS = "ABCDEFGH"


def normalize(text):
    return " ".join(re.findall(r"\w+", text.lower()))


def source():
    path = Path("data/research_external/qasc/qasc_dataset.tar.gz")
    if not path.exists():
        with urlopen(URL, timeout=30) as f:
            blob = f.read()
        if hashlib.sha256(blob).hexdigest() != SHA:
            raise ValueError("QASC publisher download changed")
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(blob)
    if digest(path) != SHA:
        raise ValueError("QASC file digest")
    with tarfile.open(path) as tar:
        return {
            split: [
                json.loads(x)
                for x in tar.extractfile("QASC_Dataset/" + split + ".jsonl")
            ]
            for split in ("train", "dev")
        }


def render(t, facts, q, options):
    user = (
        "Evidence:\n"
        + "\n".join(facts)
        + "\nQuestion: "
        + q
        + "\n"
        + "\n".join(
            f"{letter}. {v}" for letter, v in zip(LETTERS, options, strict=True)
        )
    )
    return t.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        enable_thinking=True,
        add_generation_prompt=True,
    )


def material(s):
    data = source()
    tokenizer = load_tokenizer(local_snapshot())
    used_text = set()
    used_ids = set()
    out = {}
    # Reserve the publisher development evidence first. No sentence in either
    # scored split (including distractors) can become a training example.
    for split in ("primary", "replication", "train"):
        pool = sorted(
            data["train" if split == "train" else "dev"],
            key=lambda r: fingerprint([s["seed"], split, r["id"]]),
        )
        unique = []
        for r in pool:
            keys = {
                normalize(x) for x in (r["fact1"], r["fact2"], r["question"]["stem"])
            }
            if len(keys) != 3 or keys & used_text or r["id"] in used_ids:
                continue
            used_text.update(keys)
            used_ids.add(r["id"])
            unique.append(r)
            if len(unique) == s["sizes"][split] + s["donor_pool"]:
                break
        if len(unique) != s["sizes"][split] + s["donor_pool"]:
            raise ValueError(
                "Insufficient independent question and distractor evidence"
            )
        rows = []
        for r in unique[s["donor_pool"] :]:
            donors = sorted(
                unique[: s["donor_pool"]],
                key=lambda d: fingerprint([s["seed"], r["id"], d["id"]]),
            )[:2]
            facts = [v[k] for v in (r, *donors) for k in ("fact1", "fact2")]
            facts.sort(key=lambda fact: fingerprint([s["seed"], r["id"], fact]))
            opts = sorted(
                r["question"]["choices"],
                key=lambda c: fingerprint([s["seed"], r["id"], c["label"]]),
            )
            gold = next(j for j, c in enumerate(opts) if c["label"] == r["answerKey"])
            options = [c["text"] for c in opts]
            q = r["question"]["stem"]
            text = render(tokenizer, facts, q, options)
            rows.append(
                {
                    "id": r["id"],
                    "question": q,
                    "facts": facts,
                    "options": options,
                    "answer": LETTERS[gold],
                    "donor_ids": [v["id"] for v in donors],
                    "supports": [facts.index(r[k]) for k in ("fact1", "fact2")],
                    "prompt_ids": tokenizer.encode(text, add_special_tokens=False),
                }
            )
        out[split] = rows
    return out


def answer(text, options):
    matches = re.findall(r"(?<!\w)([A-H])(?!\w)", text.strip())
    last = re.search(r"(?:answer(?: is)?|choice|option)\s*[:( ]+([A-H])\b", text, re.I)
    if last:
        return last.group(1).upper()
    if len(set(matches)) == 1:
        return matches[0]
    value = normalize(text)
    identical = [i for i, v in enumerate(options) if normalize(v) == value]
    return LETTERS[identical[0]] if len(identical) == 1 else None
