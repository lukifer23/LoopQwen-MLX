"""Reconstruct the registered batches and audit scored token classes without a model."""

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loopqwen.config import config
from loopqwen.data.sampling import prepare
from loopqwen.io import read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.evaluation.phase3 import token_classes
from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained(local_snapshot())
        self.eos_token_id = self._tokenizer.eos_token_id


def audit():
    out = Path("outputs/phase3")
    cfg = out / "configs/p3_dense.yaml"
    plan = read(cfg.with_suffix(".schedule.json"))
    c = config(cfg)
    data = prepare(c)[0]
    batches = []
    accum = Counter()
    groups = Counter()
    examples = Counter()
    exposure = {}
    hasher = hashlib.sha256()
    order = ""
    t = Tokenizer()
    for i in range(plan["250000"]["step"]):
        items = [data[(i * 4 + k) % len(data)] for k in range(4)]
        payload = [(r["id"], r["protocol"], r["prefix"], r["ids"]) for r in items]
        hasher.update(json.dumps(payload, separators=(",", ":")).encode())
        order = hashlib.sha256(
            (order + "|".join(r["id"] for r in items)).encode()
        ).hexdigest()
        batch = []
        for r in items:
            if r["protocol"] == "trace":
                tags = token_classes(r, t)
            else:
                encoded = t._tokenizer(
                    " " + r["answer"],
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                if encoded["input_ids"] + [t.eos_token_id] != r["ids"][r["prefix"] :]:
                    raise ValueError("Answer tokenization differs")
                tags = [
                    "final" if (" " + r["answer"])[a:b].strip() else "format"
                    for a, b in encoded["offset_mapping"]
                ] + ["eos"]
            accum["inputs"] += r["input_length"]
            accum["targets"] += len(tags)
            groups.update(tags)
            examples[r["protocol"]] += 1
            batch.append([r["id"], r["protocol"], r["prefix"], len(r["ids"])])
        batches.append(batch)
        for milestone, mark in plan.items():
            if mark["step"] != i + 1:
                continue
            if (
                mark["targets"] != accum["targets"]
                or mark["schedule"] != hasher.hexdigest()
            ):
                raise AssertionError("Batch or label identity differs")
            exposure[milestone] = dict(
                step=i + 1,
                **accum,
                classes=dict(groups),
                examples=dict(examples),
                schedule=hasher.hexdigest(),
                order=order,
            )
    with (out / "schedule.jsonl.gz").open("wb") as f:
        f.write(
            gzip.compress(
                (
                    "\n".join(json.dumps(b, separators=(",", ":")) for b in batches)
                    + "\n"
                ).encode(),
                mtime=0,
            )
        )
    save(out / "exposure.json", exposure)


if __name__ == "__main__":
    audit()
