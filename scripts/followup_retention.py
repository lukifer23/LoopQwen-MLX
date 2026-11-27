"""Reconstruct the three consecutive adapters' deterministic responses at the common token budget."""

import csv
from collections import Counter
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import read
from loopqwen.research.native_function_study import executions

NAMES = ("valid-return-01", "linked-rollin-02", "followup-protected-01")
FAMILIES = ("wheel", "agenda", "file", "inline", "retention")


def visible_path(request, limit):
    tr = request["trajectory"]
    turns = []
    for turn in tr["turns"]:
        tokens = turn["token_ids"]
        prefix = {"prompt": turn["prompt_ids"], "tokens": tokens[:limit]}
        if len(tokens) > limit:
            # The decode loop stops on its first stop ID. A strictly longer actual
            # turn therefore had no stop in this prefix and executes no tool here.
            turns.append({**prefix, "events": []})
            return (
                {"status": "token_limit", "answer": None, "turns": turns},
                False,
                True,
            )
        turns.append({**prefix, "events": executions({"turns": [turn]})})
    return (
        {"status": tr["status"], "answer": tr["answer"], "turns": turns},
        request["score"]["success"],
        False,
    )


def build():
    streams, panels = [], []
    for name in NAMES:
        root = Path("research/results") / name
        spec = read(root / "registration.json")
        assert spec["max_turns"] == 3
        panels.append(read(spec["material"]))
        streams.append(
            {
                a: {r["id"]: r["requests"][0] for r in unpack(root / (a + ".json.gz"))}
                for a in ("candidate", "control")
            }
        )
    older = [{w["id"]: w for w in p["rows"]} for p in panels[:2]]
    cases = [w for w in panels[2]["rows"] if w["family"] in FAMILIES]
    assert len(cases) == 36 and {w["id"] for w in cases} == set(older[0]).intersection(
        older[1]
    )
    details, counts = [], {k: Counter() for k in FAMILIES}
    limit = read("research/results/" + NAMES[-1] + "/registration.json")["max_tokens"]
    assert limit == 384
    for w in cases:
        key = w["id"]
        assert all(m[key] == w for m in older)
        if w["document"]:
            assert len({p["documents"][w["document"]] for p in panels}) == 1
        controls = [visible_path(m["control"][key], limit) for m in streams]
        assert all(z[:2] == controls[-1][:2] for z in controls)
        candidates = [visible_path(m["candidate"][key], limit) for m in streams]
        assert not controls[-1][2] and not candidates[-1][2]
        row = {
            "id": key,
            "family": w["family"],
            "base": controls[-1][1],
            "base_trace": fingerprint(controls[-1][0]),
            "earlier_base_truncated": controls[0][2],
        }
        for name, (trace, success, cutoff) in zip(NAMES, candidates, strict=True):
            row[name] = success
            row[name + "_truncated"] = cutoff
            row[name + "_trace"] = fingerprint(trace)
        details.append(row)
        g = counts[w["family"]]
        g["n"] += 1
        g["base"] += row["base"]
        for name in NAMES:
            g[name] += row[name]
        g["gained"] += row[NAMES[-1]] and not row[NAMES[-2]]
        g["lost"] += row[NAMES[-2]] and not row[NAMES[-1]]
        g["all_three"] += all(row[n] for n in NAMES)
    with Path("paper/followup-retention.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, details[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(details)
    return counts, details


if __name__ == "__main__":
    counts, _ = build()
    print({k: dict(v) for k, v in counts.items()})
