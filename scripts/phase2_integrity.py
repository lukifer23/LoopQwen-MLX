"""Replay the pinned generator and check first-response/EOS boundaries in every split."""

import json
import random
from transformers import AutoTokenizer
from loopqwen.data.reasoning import FAMILIES, problem
from loopqwen.data.v2 import ROOT, encode_protocol
from loopqwen.io import read, save
from loopqwen.model_loader import local_snapshot


def audit():
    manifest = read(ROOT / "manifest.json")
    tok = AutoTokenizer.from_pretrained(str(local_snapshot()))
    rng = random.Random(manifest["seed"])
    seen = set()
    stats = {}
    for split in ("train", "validation", "test", "ood"):
        count = 0
        labels = {"answer": 0, "trace": 0}
        with (ROOT / (split + ".jsonl")).open() as file:
            for idx, line in enumerate(file):
                r = json.loads(line)
                for _ in range(1000):
                    task = problem(
                        rng,
                        FAMILIES[idx % len(FAMILIES)],
                        idx // len(FAMILIES) % 3,
                        split == "ood",
                    )
                    if task["id"] not in seen:
                        break
                else:
                    raise AssertionError("Instance exhaustion")
                if any(r[k] != v for k, v in task.items()):
                    raise AssertionError(f"Generator mismatch {split} {idx}")
                seen.add(r["id"])
                count += 1
                for mode, value in r["variants"].items():
                    seq = value["ids"]
                    boundary = value["prefix"]
                    if (
                        seq[-1] != tok.eos_token_id
                        or len(seq) - boundary != value["output_length"]
                    ):
                        raise AssertionError("Invalid target span")
                    if (
                        idx % 47 == 0
                        and encode_protocol(task, tok, mode == "trace") != value
                    ):
                        raise AssertionError("Incorrect tokenizer segmentation")
                    labels[mode] += len(seq) - boundary
        if labels != manifest["splits"][split]["supervised"]:
            raise AssertionError("Token totals differ")
        stats[split] = {"count": count, "targets": labels}
    save("outputs/phase2/integrity.json", dict(replayed=len(seen), splits=stats))


if __name__ == "__main__":
    audit()
