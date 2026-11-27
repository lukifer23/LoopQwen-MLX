"""Trace file-value grounding, directional errors, evidence transfer and observed cost per request."""

import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_study import executions

NAME = "field-selection-01"
SPEC = Path("research/experiments") / (NAME + ".json")
STUDY = Path("research/results") / NAME
DOC = Path("docs/measurements")
PLAN = Path("research/diagnostics/field-selection-analysis.json")
ARMS = ("candidate", "control")
LABELS = ("true", "false", "uncertain")


def protocol():
    s = read(SPEC)
    p = read(s["material"])
    sample = []
    for family, key in (("inline", "tables"), ("wheel", "wheels")):
        units = [t["id"] if key == "tables" else t["package"] for t in p[key]]
        for i, unit in enumerate(sorted(units, key=lambda t: fingerprint([19013, t]))):
            view = family if family == "wheel" or i % 2 == 0 else "file"
            label = LABELS[i % (2 if family == "wheel" else 3)]
            sample.append(
                next(
                    w["id"]
                    for w in p["rows"]
                    if w["cluster"] == unit
                    and w["family"] == view
                    and w["gold"] == label
                )
            )
    data = {
        "material": digest(Path(s["material"])),
        "sample": sample,
        "reading": "One project-agent reader rates the source, final-turn generated reasoning and visible answer for one fixed request from each of the 12 tables and eight packages in both arms. Withhold model identity and scores during reading. Supported requires a correct visible verdict and an explicit, sufficient inference from the source. For uncertainty identify the specific information missing. A correct fact without the inference counts as partial. An essential false premise or wrong inference counts as incorrect. Count no visible answer or empty reasoning as absent. Mark irreducible source ambiguities separately. For cases with alternate plausible evidence record notes, without changing the fixed scorer. Blindness is limited by recognizable styles.",
        "secondary": "Recount opening calls as exact, swapped, right key and search column but different value field, other, or none. A swap holds the key text fixed and reverses both column arguments. Report subsequent genuine successful recovery following any unsuccessful first execution. Report correct verdict on an authorized successful lookup and on jointly completed items, and affirmative/negative consistency on opposite version checks of the same wheel. Report whole-table bootstrap 95% interval for the 72-question difference, seed 19014 and 20,000 draws, as descriptive uncertainty. Provide per-label confusion and the balanced constant-label ceiling with perfect retrieval. Record all output lengths, wall seconds, first latency, memory and each lost Base success. No analysis changes the primary rule.",
    }
    if PLAN.exists():
        assert data == read(PLAN)
    else:
        save(PLAN, data)
    return data


def classify(target, observed):
    if not observed:
        return "none"
    call = observed[0]["call"]
    if call == target:
        return "exact"
    a = call.get("arguments", {})
    b = target["arguments"]
    if (
        call.get("name") == target["name"]
        and a.get("document") == b["document"]
        and a.get("key") == b["key"]
    ):
        if (
            a.get("key_column") == b["value_column"]
            and a.get("value_column") == b["key_column"]
        ):
            return "swapped"
        if a.get("key_column") == b["key_column"]:
            return "other return field"
    return "other"


def extract(root):
    s = read(SPEC)
    panel = read(s["material"])
    tok = tokenizer(s)
    end = tok.convert_tokens_to_ids("</think>")
    arms = {k: unpack(root / (k + ".json.gz")) for k in ARMS}
    chosen = set(protocol()["sample"])
    rows = []
    blind = {}
    for index, w in enumerate(panel["rows"]):
        result = {k: w[k] for k in ("id", "family", "cluster", "gold")}
        q = w["questions"][0]
        result["question"] = q
        target = w["calls"][0] if w["calls"] else {}
        evidence = w.get("observation")
        for arm, stream in arms.items():
            item = stream[index]
            assert item["id"] == w["id"]
            (req,) = item["requests"]
            tr = req["trajectory"]
            tokens = tr["turns"][-1]["token_ids"]
            reasoning = tok.decode(
                tokens[: tokens.index(end)] if end in tokens else tokens
            ).strip()
            events = executions(tr)
            logical = w["family"] in ("inline", "file", "wheel")
            accurate = [
                e["call"] == target and e["observation"] == evidence for e in events
            ]
            score = req["score"]
            info = {
                "status": tr["status"],
                "answer": tr["answer"] or "",
                "reasoning": reasoning,
                "success": score["success"],
                "verdict": score.get("verdict") or "none",
                "completed": score["completed"],
                "correct": score["correct_answer"],
                "first_call": classify(target, events) if logical and target else "n/a",
                "observed": any(accurate),
                "repaired": bool(accurate and not accurate[0] and any(accurate[1:])),
                "calls": json.dumps(events, ensure_ascii=False),
                "turns": len(tr["turns"]),
                "tokens": sum(len(t["token_ids"]) for t in tr["turns"]),
                "seconds": req["wall_s"],
                "first_seconds": req["first_s"],
                "peak_bytes": req["peak"],
            }
            if w["id"] in chosen:
                source = (
                    q.partition("\n\nClaim: ")[0]
                    if w["family"] == "inline"
                    else panel["documents"][w["document"]]
                )
                if w["family"] == "file":
                    src = next(t for t in panel["tables"] if t["id"] == w["cluster"])
                    source = "\n".join(f"{k}: {v}" for k, v in src["fields"].items())
                elif w["family"] == "wheel":
                    source = (
                        w["cluster"]
                        + " — published Requires-Python "
                        + evidence["value"]
                    )
                obj = {
                    "question": q,
                    "source": source,
                    "reasoning": reasoning,
                    "answer": info["answer"],
                }
                token = fingerprint(obj)[:18]
                blind[token] = obj
                info["reading_id"] = token
            result.update({arm + "_" + k: v for k, v in info.items()})
        rows.append(result)
    return rows, dict(sorted(blind.items()))


def report(rows):
    logical = [r for r in rows if r["family"] in ("inline", "file")]
    groups = defaultdict(list)
    for r in logical:
        groups[r["cluster"]].append(r)
    ds = np.array(
        [
            sum(int(r["candidate_success"]) - int(r["control_success"]) for r in group)
            for group in groups.values()
        ]
    )
    rng = np.random.default_rng(19014)
    interval = np.quantile(
        rng.choice(ds, size=(20000, len(ds))).sum(axis=1) / 72, [0.025, 0.975]
    ).tolist()
    output = {
        "net": int(ds.sum()),
        "interval": interval,
        "constant_gold_ceiling": max(Counter(r["gold"] for r in logical).values()),
        "joint_completions": sum(
            all(r[a + "_completed"] for a in ARMS) for r in logical
        ),
        "lost_base_successes": [
            r["id"] for r in rows if r["control_success"] and not r["candidate_success"]
        ],
        "arms": {},
    }
    for arm in ARMS:

        def value(row, key, arm=arm):
            return row[arm + "_" + key]

        summary = {
            "confusion": {
                k: dict(Counter(value(r, "verdict") for r in logical if r["gold"] == k))
                for k in LABELS
            },
            "joint_correct": sum(
                value(r, "success")
                for r in logical
                if all(r[a + "_completed"] for a in ARMS)
            ),
            "families": {},
        }
        for family in ("file", "inline", "wheel", "retention"):
            subset = [r for r in rows if r["family"] == family]
            observed = [r for r in subset if value(r, "observed")]
            summary["families"][family] = {
                "grounded": sum(value(r, "success") for r in subset),
                "observed": len(observed),
                "correct_after_grounding": sum(value(r, "correct") for r in observed),
                "first_calls": dict(Counter(value(r, "first_call") for r in subset)),
                "repair": sum(value(r, "repaired") for r in subset),
                "median_seconds": float(
                    np.median([value(r, "seconds") for r in subset])
                ),
                "p90_seconds": float(
                    np.quantile([value(r, "seconds") for r in subset], 0.90)
                ),
            }
        output["arms"][arm] = summary
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["register", "blind", "report"])
    p.add_argument("--root", type=Path, default=STUDY)
    arg = p.parse_args()
    if arg.mode == "register":
        protocol()
        return
    rows, blind = extract(arg.root)
    if arg.mode == "blind":
        save(DOC / "field-selection-blinded-responses.json", blind)
        return
    outcome = report(rows)
    assert outcome["net"] == read(arg.root / "decision.json")["net"]
    ratings = read(DOC / "field-selection-blinded-ratings.json")
    assert set(ratings) == set(blind) and all(
        r["rating"] in {"supported", "partial", "incorrect", "absent", "ambiguous"}
        and r["basis"]
        for r in ratings.values()
    )
    outcome["reading"] = {
        a: dict(
            Counter(
                ratings[r[a + "_reading_id"]]["rating"]
                for r in rows
                if a + "_reading_id" in r
            )
        )
        for a in ARMS
    }
    save(DOC / "field-selection-analysis.json", outcome)
    buf = io.StringIO(newline="")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    writer = csv.DictWriter(buf, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    Path("paper/field-selection-requests.csv").write_bytes(buf.getvalue().encode())


if __name__ == "__main__":
    main()
