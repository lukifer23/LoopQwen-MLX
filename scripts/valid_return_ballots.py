"""Check whether the strict field requirement misses other complete human evidence ballots."""

import argparse
import ast
import gzip
import json
from collections import Counter
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from loopqwen.research.native_function_study import executions

PLAN = Path("research/diagnostics/valid-return-ballot-sensitivity.json")
SOURCES = Path("research/sources/valid-return/human-source-records.jsonl.gz")
SPEC = Path("research/experiments/valid-return-01.json")
ROOT = Path("research/results/valid-return-01")
OUT = Path("docs/measurements/valid-return-ballot-sensitivity.json")


def ballots():
    p = read(read(SPEC)["material"])
    source = {
        r["source_table_id"]: r
        for line in gzip.decompress(SOURCES.read_bytes()).splitlines()
        if (r := json.loads(line))["split"] != "train"
    }
    enrolled = {w["id"]: w for w in p["tables"]}
    mapping = {}
    for w in p["rows"]:
        if w["family"] != "file":
            continue
        t = source[w["cluster"]]
        row = enrolled[w["cluster"]]
        fields = [k for k in t["original_table"] if k != "title"]
        orig = next(
            q
            for q in t["claims"]
            if "\n\nClaim: " + q["hypothesis"] + "\n" in w["questions"][0]
        )
        record = orig["evidence"]
        majority = record["Majority Label"]
        majority = (
            ast.literal_eval(majority)[0] if majority.startswith("[") else majority
        )
        assert (
            majority.count("1") == 1
            and fields[majority.index("1")]
            == w["calls"][0]["arguments"]["value_column"]
        )
        masks = ast.literal_eval(record["Labels"])
        assert all(len(m) == len(fields) + 1 and set(m) <= {"0", "1"} for m in masks)
        valid = [
            frozenset(k for k, bit in zip(fields, m[:-1], strict=True) if bit == "1")
            for m in masks
            if m[-1] == "0" and "1" in m[:-1]
        ]
        mapping[w["id"]] = (row, valid, len(set(masks)) > 1)
    assert len(mapping) == 36
    return p, mapping


def register():
    _, m = ballots()
    spec = {
        "sources": digest(SOURCES),
        "panel": digest(Path(read(SPEC)["material"])),
        "requests": len(m),
        "disputed": sum(v[2] for v in m.values()),
        "protocol": "Supplemental sensitivity on 36 human file questions, fixed before the operator reads model outcomes. Re-execute the published scorer and independently identify all correctly returned fields of the intended actual source row. Each original anonymous ballot supplies a complete required field set. Exclude ballots selecting out-of-table evidence or no in-table fields. A ballot counts if all its required fields were observed. Score the same final correct completed verdict under at least one, two, or three complete ballots and under the registered majority field. Preserve every case and report any altered outcome. This exploratory test leaves the primary decision intact.",
    }
    if PLAN.exists():
        assert spec == read(PLAN)
    else:
        save(PLAN, spec)


def run(root):
    register()
    p, m = ballots()
    report = {}
    for arm in ("candidate", "control"):
        aggregate = Counter()
        cases = []
        for w, r in zip(p["rows"], unpack(root / (arm + ".json.gz")), strict=True):
            if w["id"] not in m:
                continue
            data, options, _ = m[w["id"]]
            request = r["requests"][0]
            s = request["score"]
            got = set()
            for event in executions(request["trajectory"]):
                call = event["call"]
                a = call.get("arguments", {})
                col = a.get("value_column")
                if (
                    call.get("name") == "csv_lookup"
                    and a.get("document") == w["document"]
                    and a.get("key_column") == "title"
                    and a.get("key") == data["fields"]["title"]
                    and col in data["fields"]
                    and event["observation"] == {"value": data["fields"][col]}
                ):
                    got.add(col)
            votes = sum(o <= got for o in options)
            canonical = bool(
                s["correct_answer"]
                and w["calls"][0]["arguments"]["value_column"] in got
            )
            assert canonical == s["success"]
            counts = {
                "registered": canonical,
                **{
                    f"votes_{i}": bool(s["correct_answer"] and votes >= i)
                    for i in (1, 2, 3)
                },
            }
            aggregate.update(counts)
            cases.append(
                {
                    "id": w["id"],
                    "observed_fields": sorted(got),
                    "satisfied_ballots": votes,
                    "scores": counts,
                }
            )
        report[arm] = {"totals": dict(aggregate), "cases": cases}
    if OUT.exists():
        assert read(OUT) == report
    else:
        save(OUT, report)
    print({k: v["totals"] for k, v in report.items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--register", action="store_true")
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args()
    if args.register:
        register()
    else:
        run(args.root)
