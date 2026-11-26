"""Balanced crossover for exact prefix reuse in full native tool trajectories."""

import argparse
from pathlib import Path
from statistics import median

from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.io import digest, read, save
from loopqwen.research.core import Budget
from loopqwen.research.registered import registered
from loopqwen.research.tool_fit import restore
from loopqwen.research.tool_worlds import score, tools_for
from loopqwen.tool_session import run


def signature(traj):
    return (
        traj["answer"],
        traj["status"],
        [
            (t["prompt_ids"], t["token_ids"], t.get("call"), t.get("observation"))
            for t in traj["turns"]
        ],
    )


def metrics(rows, spec):
    totals = {
        name: sum(r["trajectory"]["seconds"] for r in rows if r["policy"] == name)
        for name in ("full", "reuse")
    }
    groups = {}
    for r in rows:
        groups.setdefault(r["id"], []).append(r)
    identical = all(
        all(
            signature(v["trajectory"]) == signature(reps[0]["trajectory"]) for v in reps
        )
        for reps in groups.values()
    )
    reused = sum(
        t.get("reused_prompt_tokens", 0)
        for r in rows
        if r["policy"] == "reuse"
        for t in r["trajectory"]["turns"]
    )
    ratios = [
        median(r["trajectory"]["seconds"] for r in reps if r["policy"] == "reuse")
        / median(r["trajectory"]["seconds"] for r in reps if r["policy"] == "full")
        for reps in groups.values()
    ]
    return dict(
        passed=identical
        and reused > 0
        and totals["reuse"] <= spec["ratio"] * totals["full"],
        identical=identical,
        items=len(groups),
        totals=totals,
        reused=reused,
        median_time_ratio=median(ratios),
    )


def verify(root, decision):
    spec = read(root / "registration.json")
    rows = unpack(root / "trajectories.json.gz")
    items = {r["id"]: r for r in read(spec["material"])}
    assert len(rows) == 4 * len(spec["items"])
    for i, item in enumerate(spec["items"]):
        reps = rows[i * 4 : i * 4 + 4]
        assert all(
            r["id"] == item and score(items[item], r["trajectory"]) == r["score"]
            for r in reps
        )
        assert [r["policy"] for r in reps] == (
            ["full", "reuse", "reuse", "full"]
            if i % 2 == 0
            else ["reuse", "full", "full", "reuse"]
        )
        for r in reps:
            initial = r["trajectory"]["turns"][0]
            for t in r["trajectory"]["turns"]:
                n = t.get("reused_prompt_tokens", 0)
                assert (
                    n % 128 == 0
                    and 0 <= n < len(t["prompt_ids"])
                    and t["prompt_ids"][:n] == initial["prompt_ids"][:n]
                    and not (r["policy"] == "full" and n)
                )
    assert metrics(rows, spec) == decision


def main(name):
    spec = registered(name)
    source = Path(spec["qualification"])
    assert (
        digest(source / "checksums.json") == spec["parent_sha256"]
        and read(source / "decision.json")["passed"]
    )
    items = {r["id"]: r for r in read(spec["material"])}
    dest = Path("outputs/research") / name
    if (dest / "decision.json").exists():
        raise ValueError("Existing run")
    budget = Budget(name, spec["limits"])
    try:
        model, tok = restore(spec["model"])
        rows = []
        for i, label in enumerate(spec["items"]):
            item = items[label]
            for mode in (
                ["full", "reuse", "reuse", "full"]
                if i % 2 == 0
                else ["reuse", "full", "full", "reuse"]
            ):
                traj = run(
                    model,
                    tok,
                    item["question"],
                    tools_for(item),
                    max_tokens=128,
                    max_turns=4,
                    reuse_prefix=mode == "reuse",
                    progress=lambda _: budget.check(),
                )
                rows.append(
                    dict(
                        id=label, policy=mode, trajectory=traj, score=score(item, traj)
                    )
                )
            print(label, flush=True)
        packed(dest / "trajectories.json.gz", rows)
        save(dest / "decision.json", metrics(rows, spec))
    finally:
        budget.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    main(p.parse_args().name)
