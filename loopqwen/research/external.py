"""Pinned public development anchors. Compare completion and choice likelihood separately."""

import gc
from pathlib import Path
from time import perf_counter
import mlx.core as mx
import numpy as np
from ..io import read, save, digest
from ..data.tokenize import batch
from ..evaluation.answers import numeric
from ..evaluation.language7 import base
from ..evaluation.mechanism import packed
from ..training.losses import loss_sum
from . import core

DEMO = "Question: Four bags each contain five marbles. How many marbles are there?\nAnswer: 20\n\nQuestion: A cook has 16 eggs and uses 7. How many remain?\nAnswer: 9\n\n"


def run():
    s = read("research/experiments/external-anchor-01.json")
    out = core.OUT / s["id"]
    if (out / "decision.json").exists():
        return
    b = core.Budget(s["id"], s["limits"])
    try:
        data = read("research/data/external-anchor.json")
        boot = core.OUT / "trace-bootstrap-01"
        cp = (
            boot / "model"
            if read(boot / "decision.json")["promoted"]
            else Path("docs/measurements/phase7/transfer/model")
        )
        result = {}
        identities = {}
        for arm in ("base", "candidate"):
            if arm == "base":
                m, t, w = base()
            else:
                m, t = core.model(cp, 3)
                w = digest(cp / "adapters.safetensors")
            m.eval()
            records = []
            summ = {}
            for r in data["gsm"]:
                ids = t.encode(
                    DEMO + "Question: " + r["question"] + "\nAnswer:",
                    add_special_tokens=False,
                )
                v = core.generate(m, t, ids, 32, b.check, line=True)
                ans = numeric(v["visible"].strip()) if v["complete"] else None
                records.append(
                    dict(
                        id=r["id"],
                        family="gsm",
                        gold=r["gold"],
                        answer=ans,
                        correct=ans == numeric(r["gold"]),
                        **v,
                    )
                )
            summ["gsm"] = dict(
                n=len(records),
                correct=sum(r["correct"] for r in records),
                valid=sum(r["answer"] is not None for r in records),
                tokens=sum(len(r["token_ids"]) for r in records),
                seconds=sum(r["seconds"] for r in records),
            )
            for r in data["arc"]:
                prompt = "Question: " + r["question"].strip() + "\nAnswer:"
                totals = []
                means = []
                streams = []
                start = perf_counter()
                for choice in r["choices"]:
                    b.check()
                    d = core.doc(t, prompt, " " + choice.strip())
                    d["ids"] = d["ids"][:-1]
                    x, p, y = batch([d])
                    nll = loss_sum(m, x, p, y).item()
                    totals.append(-nll)
                    means.append(-nll / y.size)
                    streams.append(d)
                records.append(
                    dict(
                        id=r["id"],
                        family="arc",
                        gold=r["gold"],
                        streams=streams,
                        logp=totals,
                        mean=means,
                        correct=int(np.argmax(means)) == r["gold"],
                        seconds=perf_counter() - start,
                    )
                )
            arc = [r for r in records if r["family"] == "arc"]
            summ["arc"] = dict(
                n=len(arc),
                correct=sum(r["correct"] for r in arc),
                sum_correct=sum(np.argmax(r["logp"]) == r["gold"] for r in arc).item(),
                seconds=sum(r["seconds"] for r in arc),
            )
            packed(
                out / (arm + ".json.gz"),
                dict(
                    weights=w,
                    cohort=digest(Path("research/data/external-anchor.json")),
                    records=records,
                ),
            )
            from ..identity import fingerprint

            identities[arm] = fingerprint(
                [[r["id"], r.get("prompt_ids", r.get("streams"))] for r in records]
            )
            result[arm] = summ
            print(arm, summ, flush=True)
            del m, t
            gc.collect()
            mx.clear_cache()
        if len(set(identities.values())) != 1:
            raise ValueError("Unmatched external inputs")
        result["passed"] = (
            result["candidate"]["gsm"]["correct"] - result["base"]["gsm"]["correct"]
            >= 2
            and result["candidate"]["arc"]["correct"]
            >= result["base"]["arc"]["correct"] - 1
        )
        save(out / "decision.json", result)
    finally:
        b.close()


if __name__ == "__main__":
    run()
