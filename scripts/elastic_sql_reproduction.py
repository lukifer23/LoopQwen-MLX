"""Compare each already-generated native training call with the expert SQL and every available table."""

from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest
from loopqwen.research.table_worlds import canonical
from scripts.assistance_deliberation import write_csv
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

ROOT = Path("research/results/elastic-world-fit-01")
SOURCES = (
    Path("research/sources/contrastive-worlds/training.json.gz"),
    Path("research/sources/program-language/training.json.gz"),
)


def run():
    committed(ROOT / "checksums.json")
    data = {}
    all_sql = defaultdict(set)
    for source in SOURCES:
        for w in unpack(source):
            for c in w["target"].get("tool_calls", []):
                action = c["function"]
                if action["name"] == "database_query":
                    all_sql[action["arguments"]["statement"]].add(w["id"])
            if w["group"] == "operator":
                data[w["id"]] = w
    raw = ROOT / "training-free-generation.json.gz"
    out = []
    sums = Counter()
    for r in unpack(raw):
        if r["group"] != "operator":
            continue
        item = data[r["id"]]
        c = r["score"]
        answer = c["statement"]
        ref = item["target"]["tool_calls"][0]["function"]["arguments"]["statement"]
        worlds = c["worlds"]
        returns = [z["observation"].get("rows", []) for z in worlds if z["used"]]
        hits = sum(z["correct"] for z in worlds)
        values = {
            canonical(v[0][0])
            for v in returns
            if len(v) == 1 and len(v[0]) == 1 and v[0][0] is not None
        }
        out.append(
            {
                "id": r["id"],
                "expert_sql": ref,
                "generated_sql": answer,
                "exact_own_expert": answer == ref,
                "exact_any_training_target": answer in all_sql,
                "all_worlds": c["correct"],
                "tested": len(worlds),
                "correct_worlds": hits,
                "distinct_nonempty_values": len(values),
                "current_request": item["messages"][0]["content"],
            }
        )
        sums["sources"] += 1
        sums["statements"] += answer is not None
        sums["instances"] += len(worlds)
        sums["individual_correct"] += hits
        sums["partial_sources"] += 0 < hits < len(worlds)
        if c["correct"]:
            sums["whole_sources"] += 1
            sums["whole_instances"] += len(worlds)
            sums["whole_distinct_values"] += len(values) > 1
            sums["whole_exact_expert"] += answer == ref
            sums["whole_exact_any_training_target"] += answer in all_sql
    assert (sums["sources"], sums["whole_sources"]) == (24, 14)
    write_csv(Path("paper/elastic-world-sql-reproduction.csv"), out)
    report = {
        "archive": digest(ROOT / "checksums.json"),
        "streams": digest(raw),
        "training": {str(p): digest(p) for p in SOURCES},
        "counts": dict(sums),
        "interpretation": "Exact string equality against positive expert statements in both training corpora. Surface differences alone do not establish novel algorithms. The model saw all 24 questions during fitting; this distinguishes empirical recovery of enrolled programs from ability on unseen requests. All worlds were actually queried in the archived fit.",
    }
    persist(Path("docs/measurements/elastic-sql-reproduction.json"), report)
    print(dict(sums))


if __name__ == "__main__":
    run()
