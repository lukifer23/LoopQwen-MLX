"""Assess complete human SQL questions across four histories and byte-identical crossed evidence."""

import argparse
from collections import defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.research.linked_rollin import exact_sign_flip
from loopqwen.research.native_function_study import executions
from scripts.assistance_deliberation import write_csv
from scripts.register_repair_query import PANEL
from scripts.register_staged_query import STAGE
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

PLAN = Path("research/diagnostics/staged-tool-consistency.json")
CONDITIONS = ("unassisted", "valid_current", "valid_other", "provider_error")
ARMS = ("candidate", "control")


def run(freeze):
    questions = defaultdict(dict)
    schemas = defaultdict(list)
    for w in unpack(PANEL)["rows"]:
        if w["family"] == "field":
            k, condition = w["id"].rsplit("/", 1)
            assert condition == w["condition"] and condition not in questions[k]
            questions[k][condition] = w
    pairs = {}
    for k, rows in questions.items():
        assert set(rows) == set(CONDITIONS)
        schemas[rows["unassisted"]["cluster"]].append(k)
    for schema, keys in schemas.items():
        assert len(keys) == 2
        for k, other in (keys, keys[::-1]):
            a = questions[k]["valid_current"]
            b = questions[other]["valid_other"]
            assert (
                a["messages"][1:] == b["messages"][1:]
                and a["reference"] != b["reference"]
            )
            pairs[k] = {
                "second": other,
                "schema": schema,
                "history": fingerprint(a["messages"][1:]),
            }
    plan = {
        "panel": digest(PANEL),
        "primary": digest(Path("research/experiments") / (STAGE + ".json")),
        "pairs": pairs,
        "protocol": "All 24 original questions in all four supplied histories. Report a whole question only if all four registered grounded final answers succeed. Both repair means the valid-other and provider-error presentations both succeed. Selective reuse requires success on valid-current without executing another callback and a correct fresh repair on valid-other for that same question. In a crossed pair the two original questions share byte-identical supplied assistant calls and provider messages; both registered answers must succeed. Rescore each direction by the 12 original database schemas with exact paired sign flips. Report full elapsed time across the 96 requests; retain ambiguous questions and provide source identities. Descriptive secondary analysis after all masked grades are committed, with no effect on registered promotion.",
    }
    if freeze:
        committed(PANEL)
        persist(PLAN, plan)
        return
    committed(PLAN)
    assert read(PLAN) == plan
    committed(Path("docs/measurements/staged-query-reading.json"))
    archive = Path("research/results") / STAGE
    runs = {
        a: {
            r["id"]: r["requests"][0]
            for r in unpack(archive / (a + ".json.gz"))
            if r["family"] == "field"
        }
        for a in ARMS
    }
    records = []
    out = {}
    time = {a: 0.0 for a in ARMS}
    for k, items in questions.items():
        p = pairs[k]
        r = {
            "question": k,
            "schema": p["schema"],
            "wording": items["unassisted"]["interpretation"]["tag"],
            "crossed_other": p["second"],
        }
        for a in ARMS:
            rr = {c: runs[a][items[c]["id"]] for c in CONDITIONS}
            ok = {c: int(q["score"]["success"]) for c, q in rr.items()}
            r.update({a + "_" + c: v for c, v in ok.items()})
            r[a + "_all_four"] = int(all(ok.values()))
            r[a + "_both_repairs"] = ok["valid_other"] * ok["provider_error"]
            r[a + "_selective_reuse"] = int(
                ok["valid_current"]
                and not executions(rr["valid_current"]["trajectory"])
                and ok["valid_other"]
            )
            second = runs[a][questions[p["second"]]["valid_other"]["id"]]
            r[a + "_crossed"] = ok["valid_current"] * int(second["score"]["success"])
            r[a + "_wall_s"] = sum(q["wall_s"] for q in rr.values())
            time[a] += r[a + "_wall_s"]
        records.append(r)
    for key in ("all_four", "both_repairs", "selective_reuse", "crossed"):
        delta = defaultdict(int)
        for r in records:
            delta[r["schema"]] += r["candidate_" + key] - r["control_" + key]
        out[key] = {
            "n": len(records),
            **{a: sum(r[a + "_" + key] for r in records) for a in ARMS},
            "gains": sum(r["candidate_" + key] > r["control_" + key] for r in records),
            "losses": sum(r["candidate_" + key] < r["control_" + key] for r in records),
            "schema_sign_flip_p": exact_sign_flip(list(delta.values())),
        }
    write_csv(Path("paper/staged-tool-consistency.csv"), records)
    persist(
        Path("docs/measurements/staged-tool-consistency.json"),
        {
            "plan": digest(PLAN),
            "archive": digest(archive / "checksums.json"),
            "field_wall_s": time,
            "endpoints": out,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("freeze", "report"))
    run(p.parse_args().mode == "freeze")
