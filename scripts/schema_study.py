"""Counterbalance schema-guided and ordinary generation on the same recurrent weights."""

import argparse
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loopqwen.evaluation.mechanism import packed
from loopqwen.io import digest, read, save
from loopqwen.research.core import Budget
from loopqwen.research.registered import registered
from loopqwen.research.tool_fit import restore
from loopqwen.tool_session import run
from scripts.tool_study import summarize, workload


def main(name):
    spec = registered(name)
    parent = Path(spec["requires"])
    out = Path("outputs/research") / name
    if (out / "decision.json").exists():
        raise FileExistsError(out)
    if (
        digest(parent / "checksums.json") != spec["prerequisite_sha256"]
        or not read(parent / "decision.json")["passed"]
    ):
        raise ValueError("Unqualified tool weights")
    out.mkdir(parents=True, exist_ok=True)
    items, catalog, grade = workload(spec)
    arms = {"candidate": [], "control": []}
    budget = Budget(name, spec["limits"])
    try:
        model, tok = restore(parent)
        model.eval()
        for i, item in enumerate(items):
            for side in ("control", "candidate") if i % 2 else ("candidate", "control"):
                mx.reset_peak_memory()
                report = run(
                    model,
                    tok,
                    item["user_request"],
                    catalog(item),
                    max_tokens=spec["max_tokens"],
                    max_turns=spec["max_turns"],
                    schema_guided=side == "candidate",
                    progress=lambda _: budget.check(),
                )
                arms[side].append(
                    dict(id=item["id"], trajectory=report, score=grade(item, report))
                )
            if i % 8 == 7:
                print(
                    "paired",
                    i + 1,
                    [sum(r["score"]["success"] for r in a) for a in arms.values()],
                    flush=True,
                )
        for side, batch in arms.items():
            packed(out / (side + ".json.gz"), batch)
        decision = summarize(arms, spec)
        save(out / "decision.json", decision)
        print(decision, flush=True)
    finally:
        budget.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    main(p.parse_args().name)
