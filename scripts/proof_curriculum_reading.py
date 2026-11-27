"""Audit emitted evidence references, crossed judgments and a prespecified blind reading."""

import argparse
import csv
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_study import executions

SPEC = Path("research/experiments/proof-curriculum-01.json")
STUDY = Path("research/results/proof-curriculum-01")
PLAN = Path("research/diagnostics/proof-curriculum-reading-protocol.json")
LABELS = ("true", "false", "uncertain")


def prespec():
    plan = read(SPEC)
    panel = read(plan["material"])
    rows = panel["rows"]
    ordered = sorted(
        {w["cluster"] for w in rows if w["family"] == "inline"},
        key=lambda k: fingerprint([79127, k]),
    )
    items = []
    for i, key in enumerate(ordered):
        mode = ("inline", "file")[i % 2]
        items.append(
            next(
                w["id"]
                for w in rows
                if w["cluster"] == key
                and w["family"] == mode
                and w["gold"] == LABELS[i % 3]
            )
        )
    wheels = sorted({w["cluster"] for w in rows if w["family"] == "wheel"})
    for i, key in enumerate(wheels):
        items.append(
            next(
                w["id"]
                for w in rows
                if w["cluster"] == key and w["gold"] == LABELS[i % 2]
            )
        )
    assert len(set(items)) == 20
    artifact = dict(
        material=digest(Path(plan["material"])),
        sample=items,
        unit="One preselected claim for each of 12 stories and one practical request from each of eight distinct package pairs, with both arms for every item.",
        reading="One project-agent reader, blinded to arm names and raw aggregate results where feasible. For each final-turn explanation, assign supported only when it states enough correct English premises and inferences to justify its visible verdict. For uncertainty a correct explanation must identify the particular missing link or give two compatible possibilities. For packages it must use the actual constraint and correctly compare the requested version. Assign partial when a relevant true observation is given without sufficient reasoning; incorrect when an essential inference contradicts the source; absent for empty reasoning or no final verdict; ambiguous for source ambiguity that prevents judgment. Grounded tool execution is scored separately. Identical text, evidence and claim share one review.",
        secondary="Machine citation analysis on the 48 decisive FOLIO judgments in both modes. Give formal-support credit only for a completed, grounded, correct judgment whose explicit in-range premise numbers cover an inclusion-minimal sufficient subset. Record any out-of-range citation as invalid. Denominator includes replies without explicit numbered citations. Repeat separately for the seven unflagged stories. Uncertain claims are excluded from this measure. Report source-wise grounded success, completion and physical costs for the full primary stream. Resample whole stories, not individual views, for the descriptive 95% interval (20,000 draws, seed 79128). None of these analyses changes the registered decision.",
    )
    if PLAN.exists():
        assert read(PLAN) == artifact
    else:
        save(PLAN, artifact)
    return artifact


def references(text):
    hits = set()
    for bracket in re.findall(r"\[((?:\d+(?:\s*[,;]\s*\d+)*)?)\]", text):
        hits.update(map(int, re.findall(r"\d+", bracket)))
    for seq in re.findall(
        r"\b(?:premises?|statements?|rules?|facts?)\s+(\d+(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+)\d+)*)",
        text,
        re.I,
    ):
        hits.update(map(int, re.findall(r"\d+", seq)))
    return sorted(hits)


def extract(root=STUDY):
    plan = prespec()
    spec = read(SPEC)
    panel = read(spec["material"])
    minimal = {
        w["id"]: w
        for w in read("docs/measurements/proof-curriculum-minimal-support.json")["rows"]
    }
    flagged = read("research/diagnostics/reasoned-evidence-language-review.json")[
        "flagged"
    ]
    tok = tokenizer(spec)
    close = tok.convert_tokens_to_ids("</think>")
    arms = {arm: unpack(root / (arm + ".json.gz")) for arm in ("candidate", "control")}
    records = []
    samples = {}
    mapping = {}
    for i, w in enumerate(panel["rows"]):
        fam = w["family"]
        key = w["id"].rsplit("-", 1)[0]
        observations = w.get("observation")
        actual_evidence = (
            w["questions"][0].split("\n\nClaim: ")[0]
            if w["document"] is None
            else observations["value"]
            if observations and "value" in observations
            else ""
        )
        out = {k: w[k] for k in ("id", "cluster", "family", "gold", "document")}
        out.update(
            question=w["questions"][0],
            evidence=actual_evidence,
            flagged=w["cluster"] in flagged,
        )
        for arm, rows in arms.items():
            assert w["id"] == rows[i]["id"]
            q = rows[i]["requests"][0]
            tr = q["trajectory"]
            ids = tr["turns"][-1]["token_ids"]
            raw_thought = tok.decode(
                ids[: ids.index(close)] if close in ids else ids
            ).strip()
            answer = tr["answer"] or ""
            score = q["score"]
            observed = any(
                e["call"] in w["calls"] and e["observation"] == observations
                for e in executions(tr)
            )
            fields = {k: score[k] for k in ("success", "correct_answer", "completed")}
            fields.update(
                verdict=score.get("verdict"),
                observed=observed,
                reasoning=raw_thought,
                answer=answer,
                wall_s=q["wall_s"],
                first_s=q["first_s"],
                peak_bytes=q["peak"],
                tokens=sum(len(t["token_ids"]) for t in tr["turns"]),
                calls=json.dumps(executions(tr), ensure_ascii=False),
                turns=len(tr["turns"]),
            )
            if fam in ("file", "inline") and key in minimal:
                support = minimal[key]
                refs = references(raw_thought)
                valid = bool(refs) and all(1 <= v <= support["premises"] for v in refs)
                fields.update(
                    cites=json.dumps(refs),
                    sufficient=valid
                    and any(set(s).issubset(refs) for s in support["minimal"]),
                    grounded_supported=bool(
                        fields["success"]
                        and valid
                        and any(set(s).issubset(refs) for s in support["minimal"])
                    ),
                )
            if w["id"] in plan["sample"]:
                body = dict(
                    claim=w["questions"][0],
                    evidence=actual_evidence,
                    reasoning=raw_thought,
                    answer=answer,
                )
                token = fingerprint(body)[:16]
                samples[token] = body
                mapping[arm, w["id"]] = token
            out.update({arm + "_" + k: v for k, v in fields.items()})
        records.append(out)
    return records, samples, mapping


def report(records):
    outcomes = {}
    for cohort in ("all", "unflagged"):
        judged = [
            r
            for r in records
            if r["family"] in ("inline", "file")
            and (cohort == "all" or not r["flagged"])
        ]
        groups = defaultdict(list)
        for r in judged:
            groups[r["cluster"]].append(r)
        delta = np.array(
            [
                sum(int(r["candidate_success"]) - int(r["control_success"]) for r in rs)
                for rs in groups.values()
            ]
        )
        n = len(delta)
        p = (
            sum(
                abs(np.dot(delta, signs)) >= abs(delta.sum())
                for signs in itertools.product((-1, 1), repeat=n)
            )
            / 2**n
        )
        rng = np.random.default_rng(79128)
        interval = np.percentile(
            delta[rng.integers(n, size=(20000, n))].mean(axis=1) / 6, [2.5, 97.5]
        )
        info = dict(
            stories=n,
            claims=len(judged),
            net=int(delta.sum()),
            paired_p=float(p),
            difference_95_interval=interval.tolist(),
        )
        decisive = [r for r in judged if r["gold"] != "uncertain"]
        info["decisive"] = len(decisive)
        for arm in ("candidate", "control"):
            get = lambda r, k: r[arm + "_" + k]
            info[arm] = dict(
                grounded=sum(get(r, "success") for r in judged),
                fully_correct=sum(
                    all(get(r, "success") for r in rs) for rs in groups.values()
                ),
                decisive_grounded=sum(get(r, "success") for r in decisive),
                numbered_citations=sum(
                    bool(json.loads(get(r, "cites"))) for r in decisive
                ),
                sufficient_numbered_support=sum(get(r, "sufficient") for r in decisive),
                grounded_with_support=sum(
                    get(r, "grounded_supported") for r in decisive
                ),
            )
        outcomes[cohort] = info
    return outcomes


def run(mode):
    if mode == "plan":
        prespec()
        return
    rows, samples, mapping = extract()
    if mode == "blind":
        for key, body in sorted(samples.items()):
            print(json.dumps(dict(id=key, **body), ensure_ascii=False))
        return
    out = report(rows)
    dest = Path("docs/measurements/proof-curriculum-reading.json")
    if dest.exists():
        assert read(dest) == out
    else:
        save(dest, out)
    with Path("paper/proof-curriculum-requests.csv").open("w", newline="") as f:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    annotation = Path("research/diagnostics/proof-curriculum-blind-ratings.json")
    if annotation.exists():
        ratings = read(annotation)
        assert set(ratings) == set(samples)
        breakdown = defaultdict(Counter)
        for (arm, ident), key in mapping.items():
            vote = ratings[key]
            assert (
                vote["judgment"]
                in ("supported", "partial", "incorrect", "absent", "ambiguous")
                and vote["basis"]
            )
            breakdown[arm][vote["judgment"]] += 1
        save(
            "docs/measurements/proof-curriculum-blind-reading.json",
            dict(distinct=len(samples), ratings=breakdown, items=prespec()["sample"]),
        )
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "blind", "report"))
    run(parser.parse_args().mode)
