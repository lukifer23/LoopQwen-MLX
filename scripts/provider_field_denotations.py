"""Replay original released human queries on each enrolled original and published alternate database."""

import json
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest
from loopqwen.relational_tools import select
from scripts.dialogue_database_sources import CACHE, SHA
from scripts.dialogue_database_suite import SHA as SUITE_SHA
from scripts.dialogue_database_suite import SUITE
from scripts.provider_recovery_field import PANEL
from scripts.table_policy_material import persist


def source_forms(release, rows, split):
    interactions = json.loads(
        release.read(f"cosql_dataset/sql_state_tracking/cosql_{split}.json")
    )
    result = defaultdict(set)
    for dialogue in interactions:
        for turn in dialogue["interaction"]:
            result[(dialogue["database_id"], turn["utterance"])].add(turn["query"])
    return {r["id"]: sorted(result[r["source"], r["question"]]) for r in rows}


def outcomes(path, pair, forms):
    results = {}
    for r in pair:
        original = select(path, r["expert_sql"])
        assert original["referenced_tables"] and not original["truncated"]
        alternatives = forms[r["id"]]
        assert alternatives
        for query in alternatives:
            checked = select(path, query)
            assert original["rows"] == checked["rows"] and not checked["truncated"]
        results[r["id"]] = original["rows"]
    return results


def run():
    assert digest(CACHE) == SHA and digest(SUITE) == SUITE_SHA
    receipt = {"upstream": SHA, "suite": SUITE_SHA, "panels": {}}
    with (
        ZipFile(CACHE) as original,
        ZipFile(SUITE) as suite,
        TemporaryDirectory() as tmp,
    ):
        for split, panel in (
            ("dev", PANEL),
            ("train", PANEL.with_name("withheld-training-split-requests.json.gz")),
        ):
            frozen = unpack(panel)
            rows = [r for r in frozen["rows"] if r["condition"] == "unassisted"]
            forms = source_forms(original, rows, split)
            grouped = defaultdict(list)
            for r in rows:
                grouped[r["source"]].append(r)
            databases = {}
            for db, pair in sorted(grouped.items()):
                assert len(pair) == 2
                path = Path(tmp) / (db + ".sqlite")
                raw = frozen["databases"][db]
                path.write_bytes(original.read(raw["member"]))
                assert digest(path) == raw["sha256"]
                gold = outcomes(path, pair, forms)
                assert all(gold[r["id"]] == [[r["gold"]]] for r in pair)
                alternate = []
                for member in sorted(
                    n
                    for n in suite.namelist()
                    if n.startswith(f"database/{db}/") and n.endswith(".sqlite")
                ):
                    path.write_bytes(suite.read(member))
                    values = outcomes(path, pair, forms)
                    copy = Path(member).stem == db
                    if copy:
                        assert values == gold
                    alternate.append(
                        {
                            "member": member,
                            "sha256": digest(path),
                            "named_original": copy,
                            "denotations": values,
                            "distinguishes": len(
                                {json.dumps(v) for v in values.values()}
                            )
                            == 2,
                        }
                    )
                databases[db] = {
                    "original": raw,
                    "original_denotations": gold,
                    "official_sql": {r["id"]: forms[r["id"]] for r in pair},
                    "distilled": alternate,
                }
            nonoriginals = [
                r
                for d in databases.values()
                for r in d["distilled"]
                if not r["named_original"]
            ]
            changed = {
                k: sum(
                    r["denotations"][k] != baseline
                    for r in d["distilled"]
                    if not r["named_original"]
                )
                for d in databases.values()
                for k, baseline in d["original_denotations"].items()
            }
            receipt["panels"][split] = {
                "panel_sha256": digest(panel),
                "sources": len(rows),
                "by_database": databases,
                "alternate_instances": len(nonoriginals),
                "distinct_partner_results": sum(
                    r["distinguishes"] for r in nonoriginals
                ),
                "questions_with_a_changed_result": sum(
                    bool(n) for n in changed.values()
                ),
                "changed_from_original": changed,
            }
    persist(PANEL.with_name("field-denotations.json"), receipt)
    for split, data in receipt["panels"].items():
        print(
            split,
            {
                k: v
                for k, v in data.items()
                if k not in ("by_database", "changed_from_original")
            },
        )


if __name__ == "__main__":
    run()
