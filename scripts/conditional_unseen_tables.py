"""Re-execute archived model-written SQL on independently annotated extra table versions."""

import csv
import hashlib
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.table_worlds import SOURCE, prepare_tables, query_results
from scripts.assistance_deliberation import write_csv
from scripts.changed_table_sources import CACHE, FILES, REV, labels, table
from scripts.conditional_tool_transfer import P


def worlds():
    held = read(SOURCE)["cohorts"]["dev"]
    for name, (_, sha) in FILES.items():
        assert digest(CACHE / name) == sha
    ratings = labels()
    cohort = {}
    excluded = Counter()
    with ZipFile(CACHE / "original.zip") as archive:
        for w in held:
            a, b = w["versions"]
            original = archive.read(f"WikiTableQuestions-{REV}/" + w["context"])
            assert (
                table(original, ",") == a["rows"]
                and hashlib.sha256(original).hexdigest() == a["sha256"]
            )
            assert ratings[w["id"]][0]["answer"] == a["answer"]
            cohort[w["id"]] = {**w, "seen": {a["world"], b["world"]}, "versions": [a]}
    with tarfile.open(CACHE / "altered.tar.gz", mode="r|gz") as archive:
        for entry in archive:
            if len(archive.members) > 1000:
                archive.members.clear()
            if not entry.isfile():
                continue
            _, key, name = entry.name.split("/")
            if key not in cohort or not name.endswith(".tsv"):
                continue
            number = int(name[:-4])
            if not number or number not in ratings[key]:
                continue
            w = cohort[key]
            raw = archive.extractfile(entry).read()
            rows = table(raw, "\t")
            if rows is None or rows[0] != w["versions"][0]["rows"][0]:
                excluded["source shape or column names"] += 1
                continue
            if rows == w["versions"][0]["rows"]:
                excluded["identical to original table"] += 1
                continue
            w["versions"].append(
                {
                    "world": number,
                    "rows": rows,
                    **ratings[key][number],
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    for original in held:
        w = cohort[original["id"]]
        w["versions"].sort(key=lambda x: x["world"])
        by = {v["world"]: v for v in w["versions"]}
        assert len(by) == len(w["versions"]) and all(
            by[v["world"]] == v for v in original["versions"]
        )
    return cohort, dict(excluded)


def run():
    cohort, excluded = worlds()
    # Every actually generated SQL action, including repeated failures and calls
    # from both trajectories; a missing SQL action earns no source-level success.
    ledger = P / "conditional-primary-changed-statements.csv"
    with ledger.open() as f:
        calls = list(csv.DictReader(f))
    queries = defaultdict(list)
    for c in calls:
        queries[c["arm"], c["cluster"], c["sql"]].append(c)
    detail = []
    per_source = []
    with TemporaryDirectory() as t:
        folder = Path(t)
        prepare_tables(folder, cohort.values())
        for arm in ("candidate", "control"):
            for key, w in sorted(cohort.items()):
                programs = []
                for (actor, source, sql), traces in queries.items():
                    if (actor, source) != (arm, key):
                        continue
                    result = query_results(sql, w, folder)
                    by = {
                        v["world"]: r
                        for v, r in zip(w["versions"], result, strict=True)
                    }
                    for c in traces:
                        base = int(c["world"])
                        (other,) = w["seen"] - {base}
                        assert by[base]["correct"] == (
                            c["current_correct_scalar"] == "True"
                        )
                        assert by[other]["correct"] == (
                            c["same_statement_opposite_correct_scalar"] == "True"
                        )
                    enrolled = all(by[n]["correct"] for n in w["seen"])
                    full = all(r["correct"] for r in result)
                    programs.append((enrolled, full))
                    for v, r in zip(w["versions"], result, strict=True):
                        obs = r.get("observation", {})
                        detail.append(
                            {
                                "arm": arm,
                                "source": key,
                                "question": w["question"],
                                "sql": sql,
                                "original_calls": len(traces),
                                "world": v["world"],
                                "in_generation": v["world"] in w["seen"],
                                "source_sha256": v["sha256"],
                                "human_answer": v["answer"],
                                "agreeing_votes": len(v["votes"]),
                                "referenced": r["used"],
                                "correct": r["correct"],
                                "error": r.get("error"),
                                "returned": obs.get("rows"),
                                "truncated": obs.get("truncated"),
                            }
                        )
                per_source.append(
                    {
                        "arm": arm,
                        "source": key,
                        "question": w["question"],
                        "labeled_versions": len(w["versions"]),
                        "distinct_table_contents": len(
                            {fingerprint(v["rows"]) for v in w["versions"]}
                        ),
                        "additional_versions": len(w["versions"]) - 2,
                        "distinct_programs": len(programs),
                        "dual_world_programs": sum(a for a, b in programs),
                        "all_world_programs": sum(b for a, b in programs),
                        "any_dual": any(a for a, b in programs),
                        "any_all": any(b for a, b in programs),
                    }
                )
    write_csv(P / "conditional-unseen-table-executions.csv", detail)
    write_csv(P / "conditional-unseen-table-sources.csv", per_source)
    report = {
        "ledger_sha256": digest(ledger),
        "upstream": {name: sha for name, (_, sha) in FILES.items()},
        "candidate_sources": len(cohort),
        "labeled_versions": sum(len(w["versions"]) for w in cohort.values()),
        "extra_versions": sum(len(w["versions"]) - 2 for w in cohort.values()),
        "sources_with_extra": sum(len(w["versions"]) > 2 for w in cohort.values()),
        "excluded_rated_files": excluded,
        "arms": {},
    }
    for arm in ("candidate", "control"):
        ss = [r for r in per_source if r["arm"] == arm]
        ds = [r for r in detail if r["arm"] == arm]
        report["arms"][arm] = {
            "original_calls": sum(c["arm"] == arm for c in calls),
            "distinct_programs_within_source": sum(r["distinct_programs"] for r in ss),
            "reexecutions": len(ds),
            "on_extra": sum(not r["in_generation"] for r in ds),
            "full_sources_with_passing_query_on_both": sum(r["any_dual"] for r in ss),
            "full_sources_with_passing_query_on_all": sum(r["any_all"] for r in ss),
            "all_passing_sources_with_extra": [
                r["source"] for r in ss if r["any_all"] and r["additional_versions"]
            ],
            "surviving_on_extras_among_paired_wins": sum(
                r["any_all"] and r["additional_versions"] > 0 for r in ss
            ),
            "tested_on_extras_among_paired_wins": sum(
                r["any_dual"] and r["additional_versions"] > 0 for r in ss
            ),
        }
    strengths = []
    for source in sorted(report["arms"]["candidate"]["all_passing_sources_with_extra"]):
        ss = next(
            r for r in per_source if r["source"] == source and r["arm"] == "candidate"
        )
        rr = [r for r in detail if r["source"] == source and r["arm"] == "candidate"]
        passing = [
            sql
            for sql in sorted({r["sql"] for r in rr})
            if all(r["correct"] for r in rr if r["sql"] == sql)
        ]
        output = sorted(
            (r for r in rr if r["sql"] == passing[0]), key=lambda r: r["world"]
        )
        strengths.append(
            {
                "id": source,
                "question": ss["question"],
                "sql": passing[0],
                "extra": ss["additional_versions"],
                "all": ss["labeled_versions"],
                "distinct_answers": list(
                    dict.fromkeys(r["human_answer"] for r in output)
                ),
            }
        )
    report["validated_queries"] = strengths
    save(P / "conditional-unseen-tables.json", report)
    return report


if __name__ == "__main__":
    print(run())
