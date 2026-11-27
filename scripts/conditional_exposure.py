"""Independently account for complete language, delimiter and changed SQL token exposure."""

from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, save
from loopqwen.research.conditional_features import changed_token_spans
from loopqwen.research.conditional_fit import NAME, batches
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from loopqwen.research.registered import registered

OUT = Path("docs/measurements/conditional-planned-exposure.json")


def build():
    s = registered(NAME)
    tok = tokenizer(s)
    close = tok.convert_tokens_to_ids("</think>")
    stop = tok.convert_tokens_to_ids("<|im_end|>")
    rows = unpack(Path(s["training"]))
    groups = defaultdict(Counter)
    totals = Counter()
    spans = defaultdict(Counter)
    idx = []
    keys = []
    unique = set()
    targets = set()
    for r in rows:
        p, y = token_pair(tok, r)
        assert y.count(close) == y.count(stop) == 1 and y[-1] == stop
        key = fingerprint([p, y])
        unique.add(key)
        targets.add(fingerprint(y))
        end = y.index(close)
        counts = groups[r["group"]]
        rep = s["epochs"] * s["repeat_by_group"].get(r["group"], 1)
        counts["source_turns"] += 1
        counts["presentations"] += rep
        counts["distinct_position_instances"] += len(y)
        for pos, t in enumerate(y):
            if t in (close, stop):
                cat = "closing_or_stop"
            elif not tok.decode([t]).strip():
                cat = "whitespace"
            elif pos < end:
                cat = "nonwhite_deliberation"
            elif r["target"].get("tool_calls"):
                cat = "nonwhite_tool_action"
            else:
                cat = "nonwhite_visible_prose"
            counts[cat] += rep
            totals[cat] += rep
        pair = []
        for other in r.get("rejected", []):
            p2, z = token_pair(tok, {**r, "target": other["target"]})
            assert p2 == p and z[-1] == stop
            left, right = changed_token_spans(y, z)
            assert (
                y[: left[0]] == z[: right[0]]
                and y[left[-1] + 1 :] == z[right[-1] + 1 :]
            )
            a = spans[other["edit"]]
            a["pairs"] += 1
            a["chosen_positions"] += len(left)
            a["rejected_positions"] += len(right)
            pair.append({"negative_positions": right})
        idx.append({"group": r["group"], "positions": len(y), "negative": pair})
        keys.append(r["id"])
    schedule = batches(idx, s)
    exposures = Counter()
    prefs = Counter()
    for micros in schedule:
        for micro in micros:
            for i, j in micro:
                exposures[keys[i]] += 1
                if j is not None:
                    prefs[(keys[i], j)] += 1
    assert len(exposures) == len(rows) and all(
        exposures[r["id"]] == s["epochs"] * s["repeat_by_group"].get(r["group"], 1)
        for r in rows
    )
    assert (
        sum(totals.values()) == s["expected_targets"]
        and sum(prefs.values()) == s["expected_preferences"]
        and len(prefs) == 176
    )
    receipt = {
        "registration": digest(Path("research/experiments") / (NAME + ".json")),
        "training": digest(Path(s["training"])),
        "source_turns": len(rows),
        "unique_full_token_sequences": len(unique),
        "unique_assistant_token_sequences": len(targets),
        "native_updates_planned": len(schedule),
        "source_presentations": sum(exposures.values()),
        "supervised_target_positions": sum(totals.values()),
        "supervision_by_channel": totals,
        "by_group": groups,
        "complete_verified_preferences": len(prefs),
        "preference_presentations": sum(prefs.values()),
        "distinct_program_edit_spans": spans,
        "preference_presentations_per_edit_range": [
            min(prefs.values()),
            max(prefs.values()),
        ],
        "scope": "Planned exposure before optimization. Tool-action count includes syntax and arguments. Deliberation includes copied source words in older reading teachers. Whitespace decoded in isolation and closing/stop are counted separately.",
    }
    save(OUT, receipt)
    print(
        receipt["native_updates_planned"], receipt["source_presentations"], dict(totals)
    )


if __name__ == "__main__":
    build()
