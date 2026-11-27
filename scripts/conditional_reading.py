"""Keep all 156 source-visible judgments sealed before identifying either decoder."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.linked_rollin import exact_sign_flip
from loopqwen.research.table_worlds import SOURCE
from scripts.conditional_panel import FETA, NAME, NEW, PLAN
from scripts.table_policy_reading import committed

ROOT = Path("research/results") / NAME
DOC = Path("docs/measurements")
MASK = DOC / "conditional-program-blinded.json"
RATINGS = DOC / "conditional-program-blinded-ratings.json"
SCORES = DOC / "conditional-program-reading.json"
ARMS = ("candidate", "control")


def source_rows():
    panel = read(NEW)["rows"]
    by = {w["id"]: w for w in panel}
    plan = read(PLAN)
    assert plan["material"] == digest(NEW)
    changed = {w["id"]: w for w in read(SOURCE)["cohorts"]["dev"]}
    natural = {w["id"]: w for w in read(FETA / "originals.json")["test"]}
    criteria = {w["id"]: w for w in read(FETA / "source-review.json")["items"]}
    records = {}
    for key in plan["ordered_requests"]:
        w = by[key]
        f = w["family"]
        r = {"family": f, "request": w["questions"][0]}
        if f in ("natural_mounted", "natural_inline"):
            n = natural[w["row_id"]]
            r.update(
                rows=n["rows"],
                reference=n["reference"],
                source_guidance=criteria[n["id"]]["criterion"],
            )
        elif f == "changed_table":
            versions = changed[w["row_id"]]["versions"]
            (v,) = [v for v in versions if v["world"] == w["world"]]
            r.update(rows=v["rows"], reference=v["answer"])
        elif f == "database":
            r.update(reference=w["human_answer"], expected_value=w["expected_value"])
        elif f == "instruction":
            r.update(reference=w["reference"], category=w["category"])
        else:
            assert f == "open"
            r.update(reference=w["human_reply"])
        records[key] = r
    assert len(records) == 156
    return by, records


def run(mode):
    by, records = source_rows()
    left, right = [
        {r["id"]: r["requests"][0] for r in unpack(ROOT / (arm + ".json.gz"))}
        for arm in ARMS
    ]
    mapping = {}
    for k, r in records.items():
        replies = [side[k]["trajectory"]["answer"] or "" for side in (left, right)]
        distinct = sorted(set(replies), key=lambda x: fingerprint([735921, k, x]))
        r["replies"] = distinct
        mapping[k] = [distinct.index(x) for x in replies]
    if mode == "blind":
        if MASK.exists():
            assert read(MASK) == records
        else:
            save(MASK, records)
        return
    committed(MASK)
    committed(RATINGS)
    assert read(MASK) == records
    ratings = read(RATINGS)
    assert ratings.keys() == records.keys()
    summary = {a: defaultdict(Counter) for a in ARMS}
    whole = defaultdict(lambda: defaultdict(list))
    nets = defaultdict(int)
    loss = Counter()
    won = defaultdict(list)
    for k, r in records.items():
        w = by[k]
        f = r["family"]
        pair = ratings[k]
        opts = pair["replies"]
        assert len(opts) == len(r["replies"]) and pair["reason"].strip()
        assert pair["preferred"] in ["tie", *range(len(opts))]
        if len(opts) == 1:
            assert pair["preferred"] == "tie"
        usable = []
        for a, idx, q in zip(ARMS, mapping[k], (left[k], right[k]), strict=True):
            value = opts[idx]
            rating = value["rating"]
            unsupported = value["unsupported"]
            assert (
                type(unsupported) is bool
                and rating in ("fully usable", "partial", "wrong", "absent")
                and (rating == "absent") == (not r["replies"][idx].strip())
            )
            v = summary[a][f]
            v[rating] += 1
            v["unsupported"] += unsupported
            v["win"] += idx == pair["preferred"]
            v["tie"] += pair["preferred"] == "tie"
            success = rating == "fully usable" and (
                f != "natural_mounted" or q["score"]["supported_read"]
            )
            usable.append(success)
            v["grounded_usable"] += success
        loss[f] += usable == [False, True]
        nets[f] += int(usable[0]) - int(usable[1])
        won[f].append(
            (1 if pair["preferred"] == mapping[k][0] else -1)
            if pair["preferred"] != "tie"
            else 0
        )
        if f.startswith("natural_"):
            whole["natural"][w["source"]].append(usable)
        if f == "changed_table":
            whole["changed"][w["cluster"]].append(usable)
            nets["changed_" + ("original" if w["world"] == 0 else "altered")] += int(
                usable[0]
            ) - int(usable[1])
    combined = {}
    for family, groups in whole.items():
        assert len(groups) == 24 and all(len(v) == 2 for v in groups.values())
        both = [[all(j[a] for j in views) for a in (0, 1)] for views in groups.values()]
        counts = [sum(pair[i] for pair in both) for i in (0, 1)]
        combined[family] = {
            "scores": counts,
            "net": counts[0] - counts[1],
            "p": exact_sign_flip([int(a) - int(b) for a, b in both]),
        }
    ca, co = (summary[a] for a in ARMS)

    def unsupported(f):
        return ca[f]["unsupported"] <= co[f]["unsupported"]

    def paired(f):
        v = combined[f]
        return v["scores"][0] >= 6 and v["net"] >= 5 and v["p"] <= 0.05

    gates = {
        "natural": paired("natural")
        and all(
            nets[f] > 0 and not loss[f] and unsupported(f)
            for f in ("natural_mounted", "natural_inline")
        ),
        "instructions": sum(won["instruction"]) >= 6
        and ca["instruction"]["fully usable"] > co["instruction"]["fully usable"]
        and unsupported("instruction")
        and not loss["instruction"],
        "changed_table": paired("changed")
        and min(nets["changed_original"], nets["changed_altered"]) > 0
        and not loss["changed_table"]
        and unsupported("changed_table"),
        "open": ca["open"]["win"] >= max(6, co["open"]["win"] + 1)
        and nets["open"] >= 0
        and unsupported("open"),
        "database": nets["database"] >= 0,
    }
    save(
        SCORES,
        {
            "arms": summary,
            "whole": combined,
            "gates": gates,
            "passed": all(gates.values()),
            "usable_losses": loss,
            "individual_net": nets,
            "instruction_preference_p": exact_sign_flip(won["instruction"]),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["blind", "report"])
    run(parser.parse_args().mode)
