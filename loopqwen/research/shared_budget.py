"""Compare two deployments with identical self-consistency and a hard total token budget."""

import argparse
import gc
from collections import Counter
from pathlib import Path

import mlx.core as mx

from .. import REVISION
from ..deployment import load_trained
from ..evaluation.answers import numeric
from ..evaluation.language7 import base, prompt_ids
from ..evaluation.mechanism import packed
from ..evaluation.statistics import exact_mcnemar
from ..io import digest, read, save
from . import core, paired_solutions

REG = Path("research/experiments/shared-budget-vote-01.json")
DATA = Path("research/data/shared-budget-vote-01.json")


def vote(answers, remaining):
    tally = Counter(a for a in answers if a is not None)
    if not tally:
        return None, False
    ranked = tally.most_common()
    leader, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0
    return leader, top > second + remaining


def solve(m, t, item, index, s, b):
    remain = s["generation_cap"]
    traces = []
    answers = []
    ids = item["prompt_ids"]
    if prompt_ids(t, item, {"demos": 0}) != ids:
        raise ValueError("Frozen request differs")
    for j in range(s["max_attempts"]):
        seed = s["seed"] + index * s["max_attempts"] + j
        v = core.generate(
            m,
            t,
            ids,
            remain,
            b.check,
            thinking=True,
            temperature=s["temperature"] if j else 0,
            seed=seed,
        )
        del v["prompt_ids"]
        answer = numeric(v["visible"]) if v["complete"] else None
        answers.append(answer)
        traces.append(
            dict(
                **v, seed=seed, temperature=s["temperature"] if j else 0, answer=answer
            )
        )
        remain -= len(v["token_ids"])
        winner, locked = vote(answers, s["max_attempts"] - j - 1)
        if not remain or locked:
            break
    return {
        "id": item["id"],
        "prompt_ids": ids,
        "gold": item["gold"],
        "traces": traces,
        "answer": winner,
        "correct": winner == item["gold"],
        "seconds": sum(v["seconds"] for v in traces),
        "tokens": s["generation_cap"] - remain,
    }


def summarize(rows, items, s):
    report = {}
    for src in ("all", *s["sources"]):
        chosen = [i for i, r in enumerate(items) if src == "all" or r["source"] == src]
        summary = {}
        for arm in ("base", "candidate"):
            selected = [rows[arm][i] for i in chosen]
            summary[arm] = {
                "correct": sum(r["correct"] for r in selected),
                "valid": sum(r["answer"] is not None for r in selected),
                "tokens": sum(r["tokens"] for r in selected),
                "seconds": sum(r["seconds"] for r in selected),
                "attempts": sum(len(r["traces"]) for r in selected),
                "first_correct": sum(
                    r["traces"][0]["answer"] == r["gold"] for r in selected
                ),
                "first_seconds": sum(r["traces"][0]["seconds"] for r in selected),
                "fixed": sum(
                    r["correct"] and r["traces"][0]["answer"] != r["gold"]
                    for r in selected
                ),
                "lost": sum(
                    not r["correct"] and r["traces"][0]["answer"] == r["gold"]
                    for r in selected
                ),
            }
        deltas = [
            int(rows["candidate"][i]["correct"]) - int(rows["base"][i]["correct"])
            for i in chosen
        ]
        summary.update(
            wins=deltas.count(1),
            losses=deltas.count(-1),
            p=exact_mcnemar(deltas.count(1), deltas.count(-1)),
        )
        report[src] = summary
    return report


def run(freeze):
    s = read(REG)
    _, items = paired_solutions.material(
        s, read(paired_solutions.POOL) + read(paired_solutions.PANEL)
    )
    if freeze and not DATA.exists():
        save(DATA, items)
    if read(DATA) != items:
        raise ValueError("Shared-budget panel differs")
    if freeze:
        return
    if digest(DATA) != read("research/registry.json")["datasets"][str(DATA)]["sha256"]:
        raise ValueError("Unregistered panel")
    root = core.OUT / s["id"]
    if root.exists():
        raise FileExistsError(root)
    for name, k in (("parent", "parent_sha256"), ("dense_control", "dense_sha256")):
        if digest(Path(s[name]) / "adapters.safetensors") != s[k]:
            raise ValueError("Registered checkpoint identity differs")
    core.assets()
    b = core.Budget(s["id"], s["limits"])
    result = {"registration": digest(REG), "data": digest(DATA), "passed": False}
    rows = {"base": [], "candidate": []}
    try:
        half = len(items) // 2
        for start, order in ((0, ("candidate", "base")), (half, ("base", "candidate"))):
            for arm in order:
                m, t, _ = base() if arm == "base" else load_trained(s["parent"])
                for i in range(start, start + half):
                    rows[arm].append(solve(m, t, items[i], i, s, b))
                    packed(
                        root / (arm + ".json.gz"),
                        {
                            "weights": REVISION
                            if arm == "base"
                            else s["parent_sha256"],
                            "records": rows[arm],
                        },
                    )
                del m, t
                gc.collect()
                mx.clear_cache()
        scores = summarize(rows, items, s)
        pooled = scores["all"]
        result.update(
            status="completed",
            scores=scores,
            passed=pooled["p"] < 0.05
            and pooled["candidate"]["seconds"] <= pooled["base"]["seconds"]
            and all(
                scores[src]["wins"] > scores[src]["losses"]
                and scores[src]["candidate"]["correct"]
                >= scores[src]["base"]["first_correct"]
                for src in s["sources"]
            ),
        )
    except (MemoryError, TimeoutError, InterruptedError) as exc:
        result.update(status="interrupted", reason=str(exc))
    finally:
        b.close()
    save(root / "decision.json", result)
    print(result["status"], result["passed"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", action="store_true")
    run(p.parse_args().freeze)
