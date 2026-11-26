"""Three-edge CLUTRR stories with publisher kinship targets and human paraphrases."""

import ast
import csv
import itertools
import re
from pathlib import Path
from urllib.request import urlopen

from mlx_lm.utils import load_tokenizer

from ..identity import fingerprint
from ..io import digest
from ..model_loader import local_snapshot
from .deduction_data import prompt

WORDS = (
    "aunt",
    "brother",
    "daughter",
    "father",
    "granddaughter",
    "grandfather",
    "grandmother",
    "grandson",
    "mother",
    "nephew",
    "niece",
    "sister",
    "son",
    "uncle",
)
PROBE = "After considering their family story.\n</think>\n\nThe relationship is"
SHA = {
    "train": "a6123f6e305cc528f0cd9fe441014fc244cc4b4a1e13a56c381624a7ca26c392",
    "test": "e139a9ebeece6901f0aa076dd6dd86ee9ff4a750473e3d477362b7f81da359cf",
    "validation": "414390d3322b9fd27ca00d99e442df63b497c9c05918a09f72843c926556cfaa",
}
URL = "https://raw.githubusercontent.com/kliang5/CLUTRR_huggingface_dataset/e5b496941e91abb7c319d2618a3ce96752bc4ab7/gen_train234_test2to10/"


def stories(split):
    path = Path("data/research_external/clutrr") / (split + ".csv")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(URL + split + ".csv", timeout=35) as src:
            path.write_bytes(src.read())
    if digest(path) != SHA[split]:
        raise ValueError("CLUTRR conversion differs")
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            label = r["target_text"].lower()
            facts = re.split(r"(?<=[.!?])\s+", r["story"].strip())
            edges = ast.literal_eval(r["story_edges"])
            if label not in WORDS or len(facts) != 3 or len(edges) != 3:
                continue
            names = [x.split(":")[0] for x in r["genders"].split(",")]
            start, finish = ast.literal_eval(r["query_edge"])
            if (names[start], names[finish]) != ast.literal_eval(r["query"]):
                raise ValueError("Publisher graph indexing differs")
            chain = [start]
            for _ in range(3):
                linked = [
                    b if a == chain[-1] else a
                    for a, b in edges
                    if chain[-1] in (a, b) and (a not in chain or b not in chain)
                ]
                if len(linked) != 1:
                    break
                chain += linked
            if len(chain) != 4 or chain[-1] != finish:
                continue
            supports = []
            for a, b in itertools.pairwise(chain):
                hits = [
                    i
                    for i, v in enumerate(facts)
                    if "[" + names[a] + "]" in v and "[" + names[b] + "]" in v
                ]
                if len(hits) != 1:
                    break
                supports.extend(hits)
            if len(set(supports)) != 3:
                continue
            roles = {names[v]: str(i) for i, v in enumerate(chain)}
            form = [
                re.sub(
                    r"\[([^]]+)\]",
                    lambda m, roles=roles: "[" + roles.get(m[1], "other") + "]",
                    v,
                )
                .lower()
                .strip()
                for v in facts
            ]
            query = f"How is [{names[finish]}] related to [{names[start]}]?"
            yield {
                "id": r["id"],
                "story": fingerprint(sorted(facts)),
                "template": fingerprint(sorted(form)),
                "facts": facts,
                "question": query,
                "answer": label,
                "supports": supports,
            }


def material(s):
    t = load_tokenizer(local_snapshot())
    used = set()
    texts = set()
    ids = set()
    out = {}
    for name, split in (
        ("replication", "test"),
        ("primary", "validation"),
        ("train", "train"),
    ):
        rows = []
        for r in sorted(
            stories(split), key=lambda r: fingerprint([s["seed"], r["id"], r["story"]])
        ):
            if r["template"] in used or r["story"] in texts or r["id"] in ids:
                continue
            used.add(r["template"])
            texts.add(r["story"])
            ids.add(r["id"])
            rows.append(
                dict(
                    **r,
                    prompt_ids=t.encode(
                        prompt(t, r["facts"], r["question"]), add_special_tokens=False
                    ),
                )
            )
            if len(rows) == s["sizes"][name]:
                break
        if len(rows) != s["sizes"][name]:
            raise ValueError("Insufficient CLUTRR " + split + " stories")
        out[name] = rows
    if {r["answer"] for r in out["train"]} != set(WORDS):
        raise ValueError("Missing training relation")
    return out


def response(word):
    return f"I infer {word}.\n</think>\n\nThe relationship is {word}.<|im_end|>"


def answer(text, question):
    s = text.lower().replace("*", "").replace("’", "'").strip()
    aliases = {
        "mom": "mother",
        "mum": "mother",
        "dad": "father",
        "grandma": "grandmother",
        "grandpa": "grandfather",
    }
    vocab = "|".join((*WORDS, *aliases))
    explicit = re.findall(
        r"\b(?:answer|relationship)(?: is|:)\s*(?:the |a |an )?("
        + vocab
        + r")\b(?![- ]in[- ]law)",
        s,
    )
    if explicit:
        return aliases.get(explicit[-1], explicit[-1])
    target, source = [
        re.escape(x.lower()) for x in re.findall(r"\[([^]]+)\]", question)
    ]
    s = s.replace("[", "").replace("]", "")
    directed = re.findall(
        r"\b"
        + target
        + r"\s+is\s+(?:"
        + source
        + r"'s |the |a |an )?("
        + vocab
        + r")\b(?![- ]in[- ]law)",
        s,
    )
    if len(set(directed)) == 1:
        return aliases.get(directed[0], directed[0])
    if re.search(r"\b" + source + r"\s+is\s+" + target + r"'s", s) or re.search(
        r"\bnot\b", s
    ):
        return None
    found = {
        aliases.get(x, x)
        for x in re.findall(r"\b(" + vocab + r")\b(?![- ]in[- ]law)", s)
    }
    return next(iter(found)) if len(found) == 1 else None
