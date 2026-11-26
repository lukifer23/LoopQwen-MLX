"""Execute a frozen two-model tool comparison under a native research lease."""
import argparse
import gc
import gzip
import json
import math
from pathlib import Path

import mlx.core as mx

from loopqwen.deployment import load_trained
from loopqwen.evaluation.language7 import base
from loopqwen.io import read, save
from loopqwen.research.core import Budget
from loopqwen.research.tool_worlds import score, tools_for
from loopqwen.tool_session import run


def paired_p(gains, losses):
    n = gains + losses
    return min(1., 2 * sum(math.comb(n, i) for i in range(min(gains, losses)+1)) / (2**n))


def summarize(records, spec):
    metrics = {}
    for arm, rows in records.items():
        metrics[arm] = {key: sum(r["score"][key] for r in rows) for key in rows[0]["score"]}
        metrics[arm].update(seconds=sum(r["trajectory"]["seconds"] for r in rows),
                            tokens=sum(len(t["token_ids"]) for r in rows for t in r["trajectory"]["turns"]),
                            peak_bytes=max(t["peak_bytes"] for r in rows for t in r["trajectory"]["turns"]))
    pairs = list(zip(records["candidate"],records["control"],strict=True))
    gains = sum(a["score"]["success"] and not b["score"]["success"] for a,b in pairs)
    losses = sum(b["score"]["success"] and not a["score"]["success"] for a,b in pairs)
    p = paired_p(gains, losses)
    passed = (gains-losses >= spec["net_floor"] and p <= spec["p_max"] and
              metrics["candidate"]["seconds"] <= metrics["control"]["seconds"] * spec["time_ratio"])
    return dict(passed=passed,metrics=metrics,gains=gains,losses=losses,p_exact=p,items=len(pairs))


def load_arm(arm, spec, folder, budget):
    if arm == "base":
        return base()[:2]
    if arm == "release02":
        return load_trained("research/results/native-language-release-02/model")[:2]
    from loopqwen.research.tool_fit import fitted
    return fitted(arm, spec, folder, budget)


def main(name):
    spec = read(f"research/experiments/{name}.json")
    out = Path("outputs/research")/name
    if (out/"decision.json").exists():
        raise ValueError("Completed run exists")
    out.mkdir(parents=True,exist_ok=True)
    items = read(spec["material"])
    budget = Budget(name,spec["limits"])
    records = {}
    try:
        for side in ("candidate","control"):
            model, tok = load_arm(spec[side],spec,out,budget)
            mx.reset_peak_memory()
            arm = []
            for row in items:
                budget.check()
                traj = run(model,tok,row["question"],tools_for(row),max_turns=spec["max_turns"],
                           max_tokens=spec["max_tokens"],progress=lambda _:budget.check())
                arm.append(dict(id=row["id"],trajectory=traj,score=score(row,traj)))
                print(side,row["id"],traj["status"],traj["answer"],arm[-1]["score"]["success"],flush=True)
            (out/(side+".json.gz")).write_bytes(gzip.compress(json.dumps(arm).encode(),mtime=0))
            records[side] = arm
            del model,tok
            gc.collect()
            mx.clear_cache()
        assert all(a["trajectory"]["turns"][0]["prompt_ids"]==b["trajectory"]["turns"][0]["prompt_ids"]
                   for a,b in zip(records["candidate"],records["control"],strict=True))
        decision = summarize(records,spec)
        save(out/"decision.json",decision)
        print(decision,flush=True)
    finally:
        budget.close()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    main(parser.parse_args().name)
