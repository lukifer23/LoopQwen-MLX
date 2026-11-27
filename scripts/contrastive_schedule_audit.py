"""Calculate scheduled work directly from the source turns, including every negative causal prefix."""

from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest
from loopqwen.research.conditional_features import changed_token_spans
from loopqwen.research.conditional_fit import batches
from loopqwen.research.contrastive_features import FEATURES, FIT, OLD
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from loopqwen.research.registered import registered
from scripts.table_policy_material import persist


def run():
    s = registered(FIT)
    tok = tokenizer(s)
    idx = unpack(OLD / "feature-index.json.gz")
    ledger = defaultdict(Counter)
    new = []
    for w in unpack(Path(s["training"])):
        p, y = token_pair(tok, w)
        negative = []
        rec = ledger[w["group"]]
        rec["new_native_forward_calls"] += 1
        rec["new_physical_input_tokens"] += len(p) + len(y) - 1
        rec["new_stored_positions"] += len(y)
        for other in w.get("rejected", []):
            prefix, z = token_pair(tok, {**w, "target": other["target"]})
            assert prefix == p
            good, bad = changed_token_spans(y, z)
            negative.append({"positive_positions": good, "negative_positions": bad})
            rec["new_stored_positions"] += len(bad)
            if len(bad) != 1:
                rec["new_native_forward_calls"] += 1
                rec["new_physical_input_tokens"] += len(p) + len(z) - 1
        new.append(
            {
                "id": w["id"],
                "group": w["group"],
                "positions": len(y),
                "negative": negative,
            }
        )
    for name in FEATURES:
        spec = registered(name)
        idx.extend(r for r in new if r["group"] in spec["groups"])
    assert len(idx) == len({r["id"] for r in idx})
    schedule = batches(idx, s)
    largest = 0
    directions = Counter()
    for micros in schedule:
        for micro in micros:
            positions = 0
            for j, k in micro:
                r = idx[j]
                stats = ledger[r["group"]]
                stats["presentations"] += 1
                stats["supervised_positions"] += r["positions"]
                positions += r["positions"]
                if k is not None:
                    stats["preference_presentations"] += 1
                    positions += len(r["negative"][k]["negative_positions"])
                    directions[r["id"]] += 1
            largest = max(largest, positions)
    positive = sum(v["supervised_positions"] for v in ledger.values())
    prefs = sum(v["preference_presentations"] for v in ledger.values())
    assert positive == s["expected_positions"] and prefs == s["expected_preferences"]
    assert len(ledger) == 14
    out = {
        "registration": digest(Path("research/experiments") / (FIT + ".json")),
        "unique_turns": len(idx),
        "native_updates": len(schedule),
        "positive_positions": positive,
        "preference_presentations": prefs,
        "largest_microbatch_with_negative_positions": largest,
        "new_feature_forwards": sum(
            v["new_native_forward_calls"] for v in ledger.values()
        ),
        "new_feature_input_tokens": sum(
            v["new_physical_input_tokens"] for v in ledger.values()
        ),
        "feature_state_bytes_excluding_headers_labels": sum(
            v["new_stored_positions"] for v in ledger.values()
        )
        * 5
        * 1024
        * 2,
        "distinct_turns_with_preferences": len(directions),
        "groups": dict(ledger),
    }
    persist(Path("docs/measurements/contrastive-world-schedule.json"), out)
    print({k: v for k, v in out.items() if k != "groups"}, flush=True)


if __name__ == "__main__":
    run()
