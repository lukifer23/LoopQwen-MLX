"""Story-disjoint material from the published English bAbI deduction splits."""

import re
from pathlib import Path
from urllib.request import urlopen

from mlx_lm.utils import load_tokenizer

from ..identity import fingerprint
from ..io import digest, read
from ..model_loader import local_snapshot

SOURCE = "https://raw.githubusercontent.com/harvardnlp/MemN2N/f32f9531ba4230a7accc06b467c53d4d7322f62f/babi_data/en/"
DIGESTS = {
    "test": "32b77e117fe9acf5f94906684e3b5206de0108e646c76e1da3e022896ec7ee94",
    "train": "e856ebfdcce4dd7d617c1d10f985e96a0850c0ca18e9a19a017cb710d135ed3d",
}


def prompt(t, facts, question):
    content = "\n".join([*facts, question])
    return t.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def stories(split):
    path = Path("data/research_external/babi") / f"qa15_{split}.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(
            SOURCE + f"qa15_basic-deduction_{split}.txt", timeout=30
        ) as source:
            path.write_bytes(source.read())
    if digest(path) != DIGESTS[split]:
        raise ValueError("Wrong bAbI split")
    groups = {}
    for line in path.read_text().splitlines():
        n, text = line.split(" ", 1)
        if n == "1":
            numbered = {}
        if "\t" not in text:
            numbered[int(n)] = text
            continue
        question, answer, refs = text.split("\t")
        facts = list(numbered.values())
        key = fingerprint(sorted(facts))
        item = {
            "id": fingerprint([facts, question]),
            "story": key,
            "facts": facts,
            "question": question,
            "answer": answer,
            "supports": [list(numbered).index(int(i)) for i in refs.split()],
        }
        groups.setdefault(key, {})[item["id"]] = item
    return groups


def material(s):
    t = load_tokenizer(local_snapshot())
    used = {
        r["story"]
        for prev in s.get("exclude_test", [])
        for key, rows in read(prev).items()
        if key != "train"
        for r in rows
    }
    splits = {}
    for split, size in s["sizes"].items():
        groups = stories("train" if split == "train" else "test")
        keys = sorted(
            groups.keys() - used, key=lambda k: fingerprint([s["seed"], split, k])
        )
        if len(keys) < size:
            raise ValueError("Insufficient unseen stories")
        rows = []
        for k in keys[:size]:
            used.add(k)
            items = groups[k].values()
            r = min(items, key=lambda v: fingerprint([s["seed"], v["id"]]))
            rows.append(
                dict(
                    **r,
                    prompt_ids=t.encode(
                        prompt(t, r["facts"], r["question"]), add_special_tokens=False
                    ),
                )
            )
        splits[split] = rows
    return splits


def response(word):
    return f"I infer {word}.\n</think>\n\nThe animal is a {word}.<|im_end|>"


def answer(text):
    aliases = {"cats": "cat", "wolves": "wolf", "mice": "mouse"}
    # Prefer an explicit conclusion. Accept plural and singular animal names.
    last = re.split(
        r"\b(?:afraid of|animal is|final answer(?: is)?|answer is)\s*[:]?\s*",
        text.lower(),
    )[-1]
    words = [
        aliases.get(w, w)
        for w in re.findall(r"\b(?:cats?|sheep|wolves|wolf|mice|mouse)\b", last)
    ]
    return words[0] if len(set(words)) == 1 else None
