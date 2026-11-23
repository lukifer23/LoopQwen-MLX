"""Dense three-candidate plurality and greedy loop under measured sequential budgets."""

from collections import Counter
from pathlib import Path

import mlx.core as mx

from ..checkpoint import load_trained
from ..data.v2 import rows
from ..io import read, save
from .phase2 import predictions


def choose(candidates):
    votes = Counter(r["prediction"] for r in candidates if r["prediction"] is not None)
    if not votes:
        return None
    # Among equally frequent parses prefer the earliest candidate (greedy first).
    best = max(votes.values())
    return next(r["prediction"] for r in candidates if votes[r["prediction"]] == best)


def frontier(name="phase2_confirm"):
    group = read(Path("experiments") / (name + ".json"))
    docs = rows("test", "answer")[:30]
    records = []
    for dense, loop in group["matched"]:
        runs = {}
        for model_id in (loop, dense):
            m, t, state = load_trained(Path("checkpoints") / model_id)
            predictions(m, t, docs[:1], 12)
            greedy = [predictions(m, t, [r], 12) for r in docs]
            runs[model_id] = greedy
            del m
        target = sum(r["seconds"] for r in runs[loop])
        used = sum(r["seconds"] for r in runs[dense])
        greedy_seconds = used
        model, tok, state = load_trained(Path("checkpoints") / dense)
        records_i = []
        for i, r in enumerate(docs):
            base = runs[dense][i]
            candidates = [base["records"][0]]
            cost = 0
            if used < target:
                for j in (1, 2):
                    g = predictions(
                        model,
                        tok,
                        [r],
                        12,
                        0.7,
                        state["config"]["seed"] * 1000 + i * 2 + j,
                    )
                    candidates.extend(g["records"])
                    cost += g["seconds"]
            used += cost
            vote = choose(candidates)
            records_i.append(
                dict(
                    id=r["id"],
                    gold=r["answer"],
                    candidates=candidates,
                    vote=vote,
                    extra_seconds=cost,
                    loop=runs[loop][i]["records"][0],
                )
            )
        records.append(
            dict(
                seed=state["config"]["seed"],
                loop_seconds=target,
                dense_seconds=greedy_seconds,
                strategy_seconds=used,
                candidate_sets=sum(r["extra_seconds"] > 0 for r in records_i),
                dense=sum(r["candidates"][0]["correct"] for r in records_i),
                loop=sum(r["loop"]["correct"] for r in records_i),
                plurality=sum(r["vote"] == r["gold"] for r in records_i),
                records=records_i,
            )
        )
        del model
        mx.clear_cache()
    save("outputs/phase2/frontier.json", records)


if __name__ == "__main__":
    frontier()
