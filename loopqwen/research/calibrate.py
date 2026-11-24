"""Prospective first-assistant-turn calibration with a frozen untouched Base cohort."""

import random
from . import core
from ..data import graphs, reasoning
from ..evaluation.answers import numeric
from ..evaluation.language7 import base, prompt_ids
from ..evaluation.mechanism import packed
from ..identity import fingerprint
from ..io import read, save


def questions(seed, name, n):
    rooms = core.reserve(name, seed, dict(rooms=n // 4))["rooms"]
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        k = i % 4
        if k == 0:
            r = rooms[i // 4]
            q = (
                "The room links are "
                + ", ".join(f"{x} to {r['map'][x]}" for x in r["order"])
                + f". Start at room {r['start']}. Which room is reached after exactly three links?"
            )
            y = r["states"][2]
        else:
            v = (
                graphs.graph(rng, i % 2)
                if k == 3
                else reasoning.problem(
                    rng, "modular" if k == 2 else "arithmetic", i % 2
                )
            )
            q, y = v["question"], v["answer"]
        rows.append(
            dict(
                id=fingerprint(q),
                question=q,
                gold=str(y),
                kind=["room", "arithmetic", "modular", "dependency"][k],
            )
        )
    return rows


def run():
    s = read("research/experiments/turn-calibration-01.json")
    out = core.OUT / s["id"]
    if core.archived_result(out, s):
        return
    b = core.Budget(s["id"], s["limits"])
    try:
        rows = questions(s["seed"], s["id"], s["items"])
        save(out / "items.json", rows)
        m, t, w = base()
        core.assets()
        records = []
        for r in rows:
            ids = prompt_ids(t, r, s)
            result = core.generate(m, t, ids, s["cap"], b.check, thinking=True)
            answer = numeric(result["visible"]) if result["complete"] else None
            records.append(
                dict(
                    **result,
                    item=r,
                    answer=answer,
                    correct=answer == numeric(r["gold"]),
                )
            )
            packed(out / "generations.json.gz", dict(weights=w, records=records))
        valid = sum(r["answer"] is not None for r in records)
        right = sum(r["correct"] for r in records)
        save(
            out / "decision.json",
            dict(
                valid=valid,
                correct=right,
                n=len(rows),
                tokens=sum(len(r["token_ids"]) for r in records),
                seconds=sum(r["seconds"] for r in records),
                passed=valid >= 6 and 0 < right < 8,
            ),
        )
    finally:
        b.close()


if __name__ == "__main__":
    run()
