"""Intervene on two rule consequences in previously unscored bAbI stories."""

import re

from mlx_lm.utils import load_tokenizer

from ..identity import fingerprint
from ..io import read
from ..model_loader import local_snapshot
from .deduction_data import prompt


def material(s):
    t = load_tokenizer(local_snapshot())
    orig = read("research/data/deduction-stories-01.json")
    subsequent = read("research/data/deduction-stories-02.json")
    used = {v["story"] for group in subsequent.values() for v in group}
    used.update(v["story"] for group in orig.values() for v in group)
    aliases = {"wolves": "wolf", "cats": "cat", "sheep": "sheep", "mice": "mouse"}
    pat = re.compile(r"^(\w+ are afraid of )(\w+)(\.)$")
    items = []
    for row in sorted(
        orig["replication"], key=lambda v: fingerprint([s["seed"], v["id"]])
    ):
        at = row["supports"][1]
        first = pat.fullmatch(row["facts"][at])
        if not first or aliases[first[2]] != row["answer"]:
            raise ValueError("Publisher second support changed")
        for other in sorted(
            range(len(row["facts"])),
            key=lambda i: fingerprint([row["id"], s["seed"], i]),
        ):
            second = pat.fullmatch(row["facts"][other])
            if not second or first[2] == second[2]:
                continue
            changed = row["facts"].copy()
            changed[at] = first[1] + second[2] + first[3]
            changed[other] = second[1] + first[2] + second[3]
            story = fingerprint(sorted(changed))
            if story in used:
                continue
            text = prompt(t, changed, row["question"])
            ids = t.encode(text, add_special_tokens=False)
            old = row["prompt_ids"]
            if len(old) != len(ids):
                continue
            ix = [i for i, (a, b) in enumerate(zip(old, ids, strict=True)) if a != b]
            if len(ix) != 2 or [old[i] for i in ix] != [ids[i] for i in reversed(ix)]:
                continue
            used.add(story)
            key = row["id"]
            pair = [
                dict(row, pair=key, condition="original", changed_positions=ix),
                dict(
                    row,
                    pair=key,
                    condition="swapped",
                    changed_positions=ix,
                    id=fingerprint([changed, row["question"]]),
                    story=story,
                    facts=changed,
                    prompt_ids=ids,
                    answer=aliases[second[2]],
                ),
            ]
            if len(items) % 4:
                pair.reverse()
            items.extend(pair)
            break
        if len(items) == 2 * s["pairs"]:
            break
    if len(items) != 2 * s["pairs"]:
        raise ValueError("Insufficient fixed-position swaps")
    return {"intervention": items}
