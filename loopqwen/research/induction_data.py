"""Permute static premises from the published bAbI induction data; preserve proof labels."""

import re
import tarfile
from pathlib import Path
from urllib.request import urlopen

from mlx_lm.utils import load_tokenizer

from ..identity import fingerprint
from ..io import digest
from ..model_loader import local_snapshot
from .deduction_data import prompt

SHA = "84f5296ab9a1ad0dc9464e08c491d65cd08830fca3acae9ab86f75e0fb81573c"
WORDS = ("gray", "green", "white", "yellow")


def material(s):
    archive = Path("data/research_external/babi/tasks-v1.2.tar.gz")
    if not archive.exists():
        archive.parent.mkdir(exist_ok=True, parents=True)
        with urlopen(
            "https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz",
            timeout=45,
        ) as f:
            archive.write_bytes(f.read())
    if digest(archive) != SHA:
        raise ValueError("Wrong bAbI induction release")
    tok = load_tokenizer(local_snapshot())
    used = set()
    data = {}
    with tarfile.open(archive) as tar:
        for name, size in s["sizes"].items():
            split = "train" if name == "train" else "test"
            rows = {}
            for line in tar.extractfile(
                f"tasks_1-20_v1-2/en-10k/qa16_basic-induction_{split}.txt"
            ):
                n, text = line.decode().strip().split(" ", 1)
                if n == "1":
                    facts = {}
                if "\t" not in text:
                    facts[int(n)] = text
                    continue
                question, word, indices = text.split("\t")
                key = fingerprint(sorted(facts.values()))
                if key in used:
                    continue
                order = sorted(facts, key=lambda i: fingerprint([s["seed"], key, i]))
                arranged = [facts[i] for i in order]
                rows.setdefault(
                    key,
                    {
                        "id": fingerprint([arranged, question]),
                        "story": key,
                        "facts": arranged,
                        "question": question,
                        "answer": word,
                        "supports": [order.index(int(i)) for i in indices.split()],
                    },
                )
            selected = sorted(rows, key=lambda k: fingerprint([s["seed"], name, k]))[
                :size
            ]
            if len(selected) != size:
                raise ValueError("Insufficient distinct induction stories")
            data[name] = []
            for k in selected:
                row = rows[k]
                used.add(k)
                ids = tok.encode(
                    prompt(tok, row["facts"], row["question"]), add_special_tokens=False
                )
                data[name].append(dict(**row, prompt_ids=ids))
    return data


def response(word):
    return f"I infer {word}.\n</think>\n\nThe answer is {word}.<|im_end|>"


def answer(text):
    last = re.split(r"\b(?:answer is|color is|colour is)\b", text.lower())[-1].replace(
        "grey", "gray"
    )
    found = set(re.findall(r"\b(?:gray|green|white|yellow)\b", last))
    return next(iter(found)) if len(found) == 1 else None
