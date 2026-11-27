"""Compare each reply with both source contexts of its predetermined training probe."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from loopqwen.research.contrastive_fit import selection
from loopqwen.research.contrastive_grading import attested, selected_choice
from scripts.assistance_deliberation import write_csv
from scripts.program_language_material import norm

FIT = "contrastive-world-fit-01"
ROOT = Path("research/results") / FIT
PAPER = Path("paper")


def classify(reply, current, other):
    closed, text = reply["closed"], reply["visible"]
    if not closed:
        return False, False
    if current["group"] == "service_reply":

        def matches(r):
            source = json.loads(r["messages"][-1]["content"])
            return attested(text, [v.values() for v in source])

        return matches(current), matches(other)
    answers = [r["target"]["content"] for r in (current, other)]
    observed = selected_choice(text, answers)
    return observed == answers[0], observed == answers[1]


def analyze(rows, free):
    pairs = defaultdict(list)
    for row in rows:
        if row["group"] in ("qualitative", "service_reply"):
            pairs[row["group"], row["pair"]].append(row)
    records = []
    for (group, key), pair in sorted(pairs.items()):
        pair.sort(key=lambda x: x["member"])
        assert [r["member"] for r in pair] == [0, 1]
        for origin in ("original human reply", "freely generated loop"):
            if origin == "freely generated loop" and free is None:
                continue
            outputs = [
                (
                    {"closed": True, "visible": r["target"]["content"]}
                    if origin == "original human reply"
                    else free[r["id"]]
                )
                for r in pair
            ]
            shared = bool(
                norm(outputs[0]["visible"])
                and norm(outputs[0]["visible"]) == norm(outputs[1]["visible"])
            )
            for i, r in enumerate(pair):
                own, other = classify(outputs[i], r, pair[1 - i])
                if origin == "freely generated loop":
                    assert own == outputs[i]["score"]["correct"]
                records.append(
                    {
                        "id": r["id"],
                        "group": group,
                        "source_pair": key,
                        "origin": origin,
                        "closed": outputs[i]["closed"],
                        "same_nonempty_reply_in_both_contexts": shared,
                        "matches_current": own,
                        "matches_other": other,
                        "current_only": own and not other,
                        "other_only": other and not own,
                        "both": own and other,
                        "neither": not own and not other,
                    }
                )
    result = {}
    for origin in dict.fromkeys(r["origin"] for r in records):
        result[origin] = {}
        for group in ("qualitative", "service_reply"):
            rs = [r for r in records if r["origin"] == origin and r["group"] == group]
            assert len(rs) == 24
            totals = Counter()
            for r in rs:
                for k in ("current_only", "other_only", "both", "neither"):
                    totals[k] += r[k]
            totals["same_nonempty_pairs"] = (
                sum(r["same_nonempty_reply_in_both_contexts"] for r in rs) // 2
            )
            totals["both_members_current_only"] = sum(
                all(x["current_only"] for x in rs if x["source_pair"] == k)
                for k in {r["source_pair"] for r in rs}
            )
            result[origin][group] = dict(totals)
    return result, records


def run(generated=False):
    spec = read("research/experiments/" + FIT + ".json")
    rows = selection(unpack(Path(spec["training"])), spec)
    free = None
    if generated:
        free = {r["id"]: r for r in unpack(ROOT / "training-free-generation.json.gz")}
        assert {r["id"] for r in rows} == free.keys()
    result, records = analyze(rows, free)
    if generated:
        write_csv(PAPER / "contrastive-world-specificity.csv", records)
    else:
        save(
            "research/diagnostics/contrastive-fit-specificity.json",
            {
                "question": "When the model answers an enrolled probe, are the details supported specifically by the actual source rather than equally by its paired alternative? Condition on both freely generated replies and preserve all original gates. The human checks use exactly the same lexical source matcher as the frozen fit gate.",
                "fit_registration": digest(
                    Path("research/experiments") / (FIT + ".json")
                ),
                "original_human": result["original human reply"],
                "ordered": [r["id"] for r in rows],
            },
        )
    return result


if __name__ == "__main__":
    print(run())
