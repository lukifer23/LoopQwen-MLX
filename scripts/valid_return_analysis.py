"""Measure successful wrong returns, recoveries and complete multi-tool decisions from live traces."""

import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_study import executions
from scripts.field_selection_analysis import classify
from scripts.field_selection_scoring_audit import strict, tail_judgment

NAME = "valid-return-01"
ROOT = Path("research/results") / NAME
SPEC = Path("research/experiments") / (NAME + ".json")
DOC = Path("docs/measurements")
PLAN = Path("research/diagnostics/valid-return-reading.json")
ARMS = ("candidate", "control")


def prespec():
    spec = read(SPEC)
    p = read(spec["material"])
    rows = p["rows"]
    ids = []
    for fam in ("human", "wheel", "agenda"):
        selected = (
            [r for r in rows if r["family"] in ("file", "inline")]
            if fam == "human"
            else [r for r in rows if r["family"] == fam]
        )
        entities = sorted(
            {w["cluster"] for w in selected}, key=lambda k: fingerprint([427721, k])
        )
        for i, key in enumerate(entities):
            options = [w for w in selected if w["cluster"] == key]
            if fam == "human":
                req = next(
                    w
                    for w in options
                    if w["family"] == ("inline", "file")[i % 2]
                    and w["gold"] == ("true", "false", "uncertain")[i % 3]
                )
            elif fam == "wheel":
                req = next(w for w in options if w["gold"] == ("true", "false")[i % 2])
            else:
                req = next(
                    w for w in options if w["id"].endswith(("-room", "-weekday")[i % 2])
                )
            ids.append(req["id"])
    result = {
        "material": digest(Path(spec["material"])),
        "sample": ids,
        "reading": "Blind project-agent review of 28 fixed requests in both arms. Supported requires sufficient correct evidence and reasoning for the final verdict; partial names a relevant true fact without the necessary link; incorrect contains an essential error; absent lacks a final judgment or substantive reasoning; ambiguous denotes source ambiguity. Hide aggregate scores and model labels until ratings are sealed.",
        "analysis": "Replay all expected callbacks in sequence. Count correct initial field/operation, a valid but irrelevant first return (successful callback with wrong evidence), and later recovery to the full required sequence. Report each event's separate CSV and calendar milestones, opposing answers per event, class confusion, all actual fields, output lengths, per-request losses and joint lower wall, first latency and allocator peak. Compare the registered scorer with exact bare and short final-sentence verdicts. For the primary interval sample 12 whole human tables with replacement 20000 times using seed 427722. Report uncertainty and all negative results without changing the decision.",
    }
    assert len(ids) == len(set(ids)) == 28
    if PLAN.exists():
        assert result == read(PLAN)
    else:
        save(PLAN, result)
    return result


def rows_and_blind(root):
    s = read(SPEC)
    p = read(s["material"])
    selected = set(prespec()["sample"])
    traj = {k: unpack(root / (k + ".json.gz")) for k in ARMS}
    tok = tokenizer(s)
    end = tok.convert_tokens_to_ids("</think>")
    records = []
    hidden = {}
    for i, w in enumerate(p["rows"]):
        q = w["questions"][0]
        row = {k: w[k] for k in ("id", "cluster", "family", "gold")}
        row.update(
            question=q,
            required=json.dumps(w.get("ordered_events", []), ensure_ascii=False),
        )
        for arm, items in traj.items():
            src = items[i]
            assert src["id"] == w["id"]
            (req,) = src["requests"]
            tr = req["trajectory"]
            score = req["score"]
            stream = executions(tr)
            actual = tr["turns"][-1]["token_ids"]
            reason = tok.decode(
                actual[: actual.index(end)] if end in actual else actual
            ).strip()
            ans = tr["answer"] or ""
            expected = w.get("ordered_events", [])
            needs = w["document"] is not None
            known = bool(expected and stream and stream[0] == expected[0])
            successful_wrong = bool(
                expected
                and stream
                and "value" in stream[0]["observation"]
                and not known
            )
            seq = iter(stream)
            grounded = all(any(e == must for e in seq) for must in expected)
            proper = not needs or grounded
            if w["family"] != "retention":
                assert score["success"] == bool(
                    score["completed"] and proper and score.get("verdict") == w["gold"]
                )
            fields = {
                "success": score["success"],
                "complete": score["completed"],
                "verdict": score.get("verdict") or "none",
                "reason": reason,
                "answer": ans,
                "calls": json.dumps(stream, ensure_ascii=False),
                "status": tr["status"],
                "first_call": classify(expected[0]["call"], stream)
                if expected
                else "n/a",
                "first_exact": known,
                "first_valid_wrong": successful_wrong,
                "recovered": bool(successful_wrong and grounded),
                "grounded_sequence": bool(expected and grounded),
                "first_milestone": bool(expected and expected[0] in stream),
                "last_milestone": bool(expected and expected[-1] in stream),
                "bare": bool(
                    score["completed"] and proper and strict(ans) == w["gold"]
                ),
                "final_sentence": bool(
                    score["completed"] and proper and tail_judgment(ans) == w["gold"]
                ),
                "first_s": req["first_s"],
                "wall_s": req["wall_s"],
                "peak": req["peak"],
                "tokens": sum(len(v["token_ids"]) for v in tr["turns"]),
            }
            if w["id"] in selected:
                if w["family"] == "wheel":
                    evidence = (
                        w["cluster"] + " requires Python " + w["observation"]["value"]
                    )
                elif w["family"] == "agenda":
                    rec = next(x for x in p["agenda"] if x["id"] == w["cluster"])
                    evidence = f"{rec['title']}, room {rec['room']}, {rec['date']} ({date.fromisoformat(rec['date']).strftime('%A')}), Brussels local time."
                else:
                    rec = next(x for x in p["tables"] if x["id"] == w["cluster"])
                    evidence = "\n".join(f"{k}: {v}" for k, v in rec["fields"].items())
                obj = {
                    "question": q,
                    "source": evidence,
                    "reasoning": reason,
                    "answer": ans,
                }
                token = fingerprint(obj)[:18]
                hidden[token] = obj
                fields["review"] = token
            row.update({arm + "_" + k: v for k, v in fields.items()})
        records.append(row)
    return records, dict(sorted(hidden.items()))


def timing_intervals(rows):
    # Preserve the fixed mixture and keep every repeated question from one entity together.
    by_family = defaultdict(lambda: defaultdict(list))
    for row in rows:
        source = "human" if row["family"] in ("file", "inline") else row["family"]
        by_family[source][row["cluster"]].append(row)
    rng = np.random.default_rng(427723)
    total = np.zeros((20000, 2, 2))
    for group in by_family.values():
        costs = np.array(
            [
                [
                    [
                        sum(row[a + "_" + metric] for row in unit)
                        for metric in ("wall_s", "first_s")
                    ]
                    for a in ARMS
                ]
                for unit in group.values()
            ]
        )
        sampled = rng.integers(0, len(costs), size=(20000, len(costs)))
        total += costs[sampled].sum(axis=1)
    ratios = total[:, 1, :] / total[:, 0, :]
    return {
        "unit_counts": {k: len(v) for k, v in by_family.items()},
        "base_over_loop_wall": np.quantile(ratios[:, 0], [0.025, 0.975]).tolist(),
        "base_over_loop_first": np.quantile(ratios[:, 1], [0.025, 0.975]).tolist(),
    }


def report(rows, ratings):
    logical = [r for r in rows if r["family"] in ("file", "inline")]
    groups = defaultdict(list)
    for r in logical:
        groups[r["cluster"]].append(r)
    delta = [
        sum(int(r["candidate_success"]) - int(r["control_success"]) for r in group)
        for group in groups.values()
    ]
    rng = np.random.default_rng(427722)
    ci = np.quantile(
        rng.choice(delta, (20000, len(delta))).sum(axis=1) / len(logical),
        [0.025, 0.975],
    ).tolist()
    results = {
        "human_net": sum(delta),
        "timing_intervals": timing_intervals(rows),
        "human_cluster_interval": ci,
        "lost": [
            r["id"] for r in rows if r["control_success"] and not r["candidate_success"]
        ],
        "gained": [
            r["id"] for r in rows if not r["control_success"] and r["candidate_success"]
        ],
        "arms": {},
    }
    results["jointly_faster_gains"] = [
        r["id"]
        for r in rows
        if r["id"] in results["gained"]
        and all(
            r["candidate_" + k] < r["control_" + k]
            for k in ("wall_s", "first_s", "peak")
        )
    ]
    for a in ARMS:
        arm = {}
        for f in ("inline", "file", "wheel", "agenda", "retention"):
            subset = [r for r in rows if r["family"] == f]
            arm[f] = {
                k: sum(r[a + "_" + k] for r in subset)
                for k in (
                    "success",
                    "complete",
                    "first_exact",
                    "first_valid_wrong",
                    "recovered",
                    "grounded_sequence",
                    "first_milestone",
                    "last_milestone",
                    "bare",
                    "final_sentence",
                )
            }
            arm[f]["opening"] = dict(Counter(r[a + "_first_call"] for r in subset))
        arm["rating"] = dict(
            Counter(
                ratings[r[a + "_review"]]["rating"] for r in rows if a + "_review" in r
            )
        )
        arm["human_confusion"] = {
            k: dict(Counter(r[a + "_verdict"] for r in logical if r["gold"] == k))
            for k in ("true", "false", "uncertain")
        }
        results["arms"][a] = arm
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("register", "blind", "report"))
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args()
    if args.mode == "register":
        prespec()
        return
    records, blind = rows_and_blind(args.root)
    if args.mode == "blind":
        save(DOC / "valid-return-blinded-responses.json", blind)
        return
    ratings = read(DOC / "valid-return-blinded-ratings.json")
    assert set(ratings) == set(blind) and all(
        r["rating"] in {"supported", "partial", "incorrect", "absent", "ambiguous"}
        and r["basis"]
        for r in ratings.values()
    )
    result = report(records, ratings)
    assert result["human_net"] == read(args.root / "decision.json")["net"]
    save(DOC / "valid-return-analysis.json", result)
    dest = io.StringIO(newline="")
    writer = csv.DictWriter(
        dest, fieldnames=list(dict.fromkeys(k for r in records for k in r))
    )
    writer.writeheader()
    writer.writerows(records)
    Path("paper/valid-return-requests.csv").write_bytes(dest.getvalue().encode())


if __name__ == "__main__":
    main()
