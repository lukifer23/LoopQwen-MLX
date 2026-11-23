"""Reconstruct complete logical schedules and audit token/weight accounting."""

import gzip
import json
import hashlib
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loopqwen.config import config
from loopqwen.data.sampling import prepare
from loopqwen.data.token_classes import token_classes
from loopqwen.io import read, save, checkpoint_location
from loopqwen.model_loader import local_snapshot
from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained(local_snapshot())
        self.eos_token_id = self._tokenizer.eos_token_id


def run():
    out = Path("outputs/phase4")
    tok = Tokenizer()
    summary = {}
    for path in sorted((out / "configs").glob("*.yaml")):
        if path.stem.startswith("p4_profile"):
            continue
        c = config(path)
        gen = checkpoint_location(Path("checkpoints") / path.stem)
        state = read(gen / "state.json")
        if state["status"] != "completed":
            raise ValueError("Unfinished run " + path.stem)
        pool = prepare(c)[0]
        schedule = hashlib.sha256()
        order = ""
        protocols = Counter()
        target = Counter()
        inputs = Counter()
        classes = Counter()
        effective = Counter()
        cursor = 0
        ids = set()
        with gzip.GzipFile(out / (path.stem + ".schedule.gz"), "wb", mtime=0) as stream:
            for step, h in enumerate(state["history"]):
                docs = [
                    pool[(cursor + i) % len(pool)]
                    for i in range(c["batch_size"] * c["accumulation"])
                ]
                chunk = []
                logical_labels = 0
                weight = 0.0
                for r in docs:
                    cl = token_classes(r, tok)
                    if len(cl) != r["output_length"]:
                        raise ValueError("Bad class alignment")
                    name = r["protocol"]
                    w = sum(c.get("token_weights", {}).get(v, 1.0) for v in cl)
                    protocols[name] += 1
                    target[name] += len(cl)
                    inputs[name] += r["input_length"]
                    effective[name] += w
                    weight += w
                    classes.update(cl)
                    logical_labels += len(cl)
                    ids.add(r["id"])
                    chunk.append([r["id"], name, r["prefix"], r["ids"]])
                cursor += len(docs)
                if (
                    logical_labels != h["labels"]
                    or abs(weight - h["objective_denominator"]) > 1e-5
                ):
                    raise ValueError("Logical denominator mismatch")
                order = hashlib.sha256(
                    (order + "|".join(r["id"] for r in docs)).encode()
                ).hexdigest()
                line = json.dumps(chunk, separators=(",", ":")).encode() + b"\n"
                schedule.update(line)
                stream.write(line)
        if (order, sum(target.values()), sum(inputs.values()), cursor) != (
            state["order_hash"],
            state["loss_tokens"],
            state["input_tokens"],
            state["cursor"],
        ):
            raise ValueError(
                "Checkpoint exposure disagrees with independent schedule reconstruction"
            )
        summary[path.stem] = dict(
            step=state["step"],
            unique=len(ids),
            repeats=cursor - len(ids),
            examples=protocols,
            targets=target,
            inputs=inputs,
            weighted_targets=effective,
            classes=classes,
            schedule=schedule.hexdigest(),
            order=order,
        )
    save(out / "exposure.json", summary)


if __name__ == "__main__":
    run()
