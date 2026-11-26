"""Change the queried individual's class with one token exchange on held-out stories."""

import re

from mlx_lm.utils import load_tokenizer

from ..identity import fingerprint
from ..io import read
from ..model_loader import local_snapshot
from . import induction_data
from .deduction_data import prompt
from .induction_data import bindings


def material(spec):
    t = load_tokenizer(local_snapshot())
    reserved = induction_data.material(dict(spec, sizes={"primary": 500}))["primary"]
    used = {
        r["story"]
        for file in spec["exclude"]
        for split in read(file).values()
        for r in split
    }
    reserved_stories = {r["story"] for r in reserved}
    result = []
    for row in reserved:
        original = row["facts"]
        index, evidence = bindings(original, row["question"])
        if len(evidence) != 1 or evidence[0][2] != row["answer"]:
            continue
        for other in sorted(
            range(len(original)), key=lambda i: fingerprint([row["id"], i])
        ):
            if not re.fullmatch(r"\w+ is an? \w+\.", original[other]):
                continue
            changed = original.copy()
            # Exchange only the class names, retaining each sentence's subject and article.
            left, old = original[index].rsplit(" ", 1)
            right, new = original[other].rsplit(" ", 1)
            changed[index], changed[other] = left + " " + new, right + " " + old
            key = fingerprint(sorted(changed))
            if key in used or key in reserved_stories:
                continue
            _, after = bindings(changed, row["question"])
            if len(after) != 1 or after[0][2] == row["answer"]:
                continue
            peer, color, answer = after[0]
            ids = t.encode(
                prompt(t, changed, row["question"]), add_special_tokens=False
            )
            before = row["prompt_ids"]
            if len(ids) != len(before):
                continue
            ix = [i for i, (a, b) in enumerate(zip(ids, before, strict=True)) if a != b]
            if len(ix) != 2 or [before[i] for i in ix] != [
                ids[i] for i in reversed(ix)
            ]:
                continue
            used.add(key)
            args = {"pair": row["id"], "changed_positions": ix}
            pair = [
                dict(row, condition="original", **args),
                dict(
                    row,
                    id=fingerprint([changed, row["question"]]),
                    story=key,
                    facts=changed,
                    supports=[index, peer, color],
                    answer=answer,
                    prompt_ids=ids,
                    condition="swapped",
                    **args,
                ),
            ]
            if len(result) % 4:
                pair.reverse()
            result.extend(pair)
            break
        if len(result) == 2 * spec["pairs"]:
            break
    if len(result) != 2 * spec["pairs"]:
        raise ValueError("Insufficient unique, unambiguous class swaps")
    return {"intervention": result}
