"""Identify free strict gains in exchanges where both decoders first requested the same operation."""

import csv
from collections import Counter

from loopqwen.evaluation.mechanism import unpack
from loopqwen.research.native_function_study import executions


def write(root, path):
    arms = [unpack(root / (s + ".json.gz")) for s in ("candidate", "control")]
    counts = Counter()
    with path.open("w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "family",
                "initial_action_relationship",
                "opening_tokens_identical",
                "loop_correct",
                "base_correct",
            ]
        )
        for a, b in zip(*arms, strict=True):
            assert a["id"] == b["id"]
            # Only first user exchanges have exogenous identical initial histories.
            c, d = (r["requests"][0] for r in (a, b))
            x, y = (q["trajectory"]["turns"][0] for q in (c, d))
            u, v = ([z["call"] for z in executions({"turns": [t]})] for t in (x, y))
            kind = (
                "same executed call(s)"
                if u and u == v
                else "both without execution"
                if not u and not v
                else "different first action"
            )
            ac, bc = c["score"]["success"], d["score"]["success"]
            counts[kind, "n"] += 1
            counts[kind, "win"] += int(ac and not bc)
            counts[kind, "loss"] += int(bc and not ac)
            counts[kind, "tokens_match"] += int(x["token_ids"] == y["token_ids"])
            w.writerow(
                [a["id"], a["family"], kind, x["token_ids"] == y["token_ids"], ac, bc]
            )
    return counts
