"""Separate stale context, genuine progress and human answer variation in paired continuations."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.native_function_study import executions

PANEL = Path("research/data/followup-protected-panel.json")
PLAN = Path("research/diagnostics/followup-reading.json")
ROOT = Path("research/results/followup-protected-01")
DOC = Path("docs/measurements")
BLIND = DOC / "followup-blinded-answers.json"
RATINGS = DOC / "followup-blinded-ratings.json"
ARMS = ("candidate", "control")
VERDICTS = ("supported", "partial", "wrong", "absent", "justified abstention")


def plan(rows):
    human = [r for r in rows if r["family"].startswith("human_")]
    titles = sorted(
        {r["cluster"] for r in human}, key=lambda c: fingerprint([740108, c])
    )
    samples = [
        next(r["id"] for r in human if r["cluster"] == c and r["turn"] == i % 2)
        for i, c in enumerate(titles)
    ]
    assert len(set(samples)) == 24
    return {
        "panel": digest(PANEL),
        "human_cases": samples,
        "verdicts": VERDICTS,
        "review": "One actual answer from each arm for one question per held title, alternating first and second questions in seeded title order. Match against the complete request and published annotations with model identities and scores hidden. Rate absent, supported, partial, wrong, or justified abstention. Separately mark a substantive unsupported claim. Seal before reading primary scores. Exact duplicate answers on a question receive one rating.",
    }


def cases(rows, selected):
    raw = {a: unpack(ROOT / (a + ".json.gz")) for a in ARMS}
    assert all(len(v) == len(rows) for v in raw.values())
    masked, pairs = {}, []
    for i, w in enumerate(rows):
        out = {"id": w["id"], "family": w["family"], "cluster": w["cluster"]}
        for arm in ARMS:
            case = raw[arm][i]
            assert case["id"] == w["id"]
            (r,) = case["requests"]
            tr = r["trajectory"]
            ans = tr["answer"] or ""
            calls = executions(tr)
            fields = {
                **r["score"],
                "status": tr["status"],
                "answer": ans,
                "calls": json.dumps(calls, ensure_ascii=False),
                "tokens": sum(len(t["token_ids"]) for t in tr["turns"]),
                "first_s": r["first_s"],
                "wall_s": r["wall_s"],
                "peak": r["peak"],
            }
            if w["id"] in selected:
                ob = {
                    "request": w["questions"][0],
                    "human_annotations": w["annotations"],
                    "visible_answer": ans,
                }
                key = fingerprint(ob)[:20]
                assert key not in masked or masked[key] == ob
                masked[key] = ob
                fields["blind_key"] = key
            if w["family"] == "archive":
                for label in ("current", "previous"):
                    expected = w[label]
                    fields[label + "_read"] = any(
                        e["call"]["name"] == expected["call"]["name"]
                        and e["call"]["arguments"].get("wheel")
                        == expected["call"]["arguments"]["wheel"]
                        and str(e["call"]["arguments"].get("field", "")).casefold()
                        == "requires-python"
                        and e["observation"] == expected["observation"]
                        for e in calls
                    )
            out.update({arm + "_" + k: v for k, v in fields.items()})
        pairs.append(out)
    return pairs, dict(sorted(masked.items()))


def run(mode):
    rows = read(PANEL)["rows"]
    selection = plan(rows)
    if mode == "plan":
        save(PLAN, selection)
        return
    assert json.loads(json.dumps(selection)) == read(PLAN)
    pairs, masked = cases(rows, selection["human_cases"])
    if mode == "blind":
        if BLIND.exists():
            assert read(BLIND) == masked
        else:
            save(BLIND, masked)
        return
    assert read(BLIND) == masked
    rates = read(RATINGS)
    assert set(rates) == set(masked)
    summary = {}
    for arm in ARMS:
        hist = Counter()
        for r in pairs:
            k = r.get(arm + "_blind_key")
            if k is not None:
                v = rates[k]
                assert v["verdict"] in VERDICTS and type(v["unsupported_claim"]) is bool
                hist[v["verdict"]] += 1
                hist["unsupported_claim"] += v["unsupported_claim"]
        results = {}
        for fam in ("human_first", "human_followup", "archive"):
            group = [r for r in pairs if r["family"] == fam]
            fields = (
                (
                    "success",
                    "current_read",
                    "previous_read",
                    "measured",
                    "used_older_constraint",
                    "constraint",
                    "verdict",
                )
                if fam == "archive"
                else (
                    "success",
                    "previous_repeated",
                    "human_equivalent",
                    "human_agreement",
                    "f1",
                )
            )
            results[fam] = {k: sum(r[arm + "_" + k] for r in group) for k in fields}
            results[fam]["requests"] = len(group)
        summary[arm] = {"masked_review": dict(hist), "observations": results}
    for fam in sorted({r["family"] for r in pairs}):
        group = [r for r in pairs if r["family"] == fam]
        summary.setdefault("transitions", {})[fam] = dict(
            Counter(
                ("both" if r["control_success"] else "gained")
                if r["candidate_success"]
                else ("lost" if r["control_success"] else "neither")
                for r in group
            )
        )
    save(DOC / "followup-reading.json", summary)
    columns = list(dict.fromkeys(k for r in pairs for k in r))
    with Path("paper/followup-requests.csv").open("w", newline="") as f:
        out = csv.DictWriter(f, columns, lineterminator="\n")
        out.writeheader()
        out.writerows(pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "blind", "report"))
    run(parser.parse_args().action)
