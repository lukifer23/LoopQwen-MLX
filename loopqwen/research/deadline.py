"""Train an answer-only recurrent exit for a nearly exhausted native reasoning turn."""

import argparse
from pathlib import Path

from mlx_lm.utils import load_tokenizer

from ..evaluation.mechanism import unpack
from ..identity import fingerprint
from ..io import digest, read, save
from ..model_loader import local_snapshot
from . import core, deliberation, paired_solutions

REG = Path("research/experiments/late-recurrent-exit-01.json")
PANEL = Path("research/data/late-recurrent-exit-01.json")
TRAIN = Path("research/data/late-recurrent-exit-training.json")


def material(s):
    prior = [x for path in s["excluded_panels"] for x in read(path)]
    used = {i for p in s["excluded_training"] for i in read(p)["ordered_questions"]}
    _, rows = paired_solutions.material(s, prior, exclude_ids=used)
    drafts = Path(s["drafts"])
    if digest(drafts) != s["draft_sha256"]:
        raise ValueError("Original training-only drafts changed")
    originals = {i["id"]: i for i in read(deliberation.TRAIN)}
    groups = {True: [], False: []}
    for d in unpack(drafts):
        groups[d["close_at"] is None].append(d)
    for v in groups.values():
        v.sort(key=lambda d: fingerprint([s["seed"], d["id"]]))
    t = load_tokenizer(local_snapshot())
    docs = []
    keys = []
    for pair in zip(groups[True][:16], groups[False][:16], strict=True):
        for d in pair:
            item = originals[d["id"]]
            if d["prompt_ids"] != item["prompt_ids"]:
                raise ValueError("Draft identity")
            upto = (
                d["close_at"]
                if d["close_at"] is not None
                else len(d["token_ids"]) - int(d["stop"] != "cap")
            )
            prefix = d["prompt_ids"] + d["token_ids"][:upto]
            suffix = t.encode(
                "</think>\n\nFinal answer: " + item["gold"], add_special_tokens=False
            ) + [core.atom(t, "<|im_end|>")]
            docs.append({"ids": prefix + suffix, "prefix": len(prefix)})
            keys.append(d["id"])
    plan = {
        "ordered_questions": keys,
        "documents": fingerprint(docs),
        "targets": sum(len(d["ids"]) - d["prefix"] for d in docs),
    }
    return docs, rows, plan


def run(freeze):
    s = read(REG)
    docs, rows, plan = material(s)
    for p, value in ((TRAIN, plan), (PANEL, rows)):
        if freeze and not p.exists():
            save(p, value)
        if read(p) != value:
            raise ValueError("Frozen exit material differs")
    if not freeze:
        deliberation.compare(s, REG, PANEL, TRAIN, rows, lambda budget, root: docs)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", action="store_true")
    run(p.parse_args().freeze)
