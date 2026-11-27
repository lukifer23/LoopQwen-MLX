"""Measure which human-supported service fields are actually new in the tool result."""

from collections import Counter

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest
from scripts.contrastive_world_material import OUT, SERVICES
from scripts.program_language_material import norm
from scripts.table_policy_material import persist


def evaluate(pool):
    counts = Counter()
    novel = Counter()
    outside = Counter()
    for w in pool:
        past = " " + norm(" ".join(m["content"] for m in w["messages"])) + " "
        # Select the first certified single-row support, as in the independent incompatible-pair audit.
        support = w["supports"][0]
        row = w["observation"][support["result_index"]]
        values = {norm(v) for v in support["fields"].values()}
        assert all(norm(row[k]) == norm(v) for k, v in support["fields"].items())
        absent = {v for v in values if " " + v + " " not in past}
        assert absent
        counts[len(values)] += 1
        novel[len(absent)] += 1
        for k, v in support["fields"].items():
            if norm(v) in absent:
                outside[k] += 1
    return {
        "human_responses": len(pool),
        "actual_single_row_support_values": dict(sorted(counts.items())),
        "supported_values_absent_from_prior_conversation": dict(sorted(novel.items())),
        "at_least_two_new_values": sum(n for k, n in novel.items() if k >= 2),
        "new_supported_value_fields": dict(sorted(outside.items())),
    }


def run():
    path = OUT / "training.json.gz"
    held = OUT / "withheld.json.gz"
    rows = unpack(path)
    original = {r["id"]: r for r in unpack(SERVICES / "training.json.gz")}
    ids = [r["pair"] for r in rows if r["group"] == "service_action"]
    assert len(ids) == len(set(ids)) == 1800
    train = evaluate([original[k] for k in ids])
    test = evaluate([w for p in unpack(held)["services"] for w in p])
    result = {
        "training": digest(path),
        "withheld": digest(held),
        "train": train,
        "reserved": test,
        "interpretation": "The source response names these original field values verbatim. A value absent from the dialogue cannot be copied directly from its text. Prior knowledge and correlations between a request and plausible results can still support a guess. The two service dialogues in a pair may have different histories. Only the table pairs repeat identical user openings with changed human answers.",
    }
    persist(OUT / "service-evidence-audit.json", result)
    print(train["at_least_two_new_values"], test["at_least_two_new_values"])


if __name__ == "__main__":
    run()
