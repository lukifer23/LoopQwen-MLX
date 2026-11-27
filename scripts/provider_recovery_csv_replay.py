"""Replay crossed capitalization and query repair on every independently labeled table."""

import csv
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen.csv_tools import tools as csv_tools
from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.relational_tools import tools
from loopqwen.research.table_worlds import FILENAME, canonical, mount
from loopqwen.tool_session import compact
from scripts.provider_recovery_csv import ENROLLED, MULTI
from scripts.provider_recovery_material import ROOT
from scripts.provider_recovery_replay import session
from scripts.table_policy_material import persist


def replay():
    target = ROOT / "csv-training.json.gz"
    info = read(ROOT / "csv-qualification.json")
    assert (digest(target), digest(MULTI), digest(ENROLLED)) == (
        info["training"],
        info["original_tables"],
        info["eligible_training_cohort"],
    )
    originals = {w["id"]: w for w in unpack(MULTI)}
    eligible = {w["pair"] for w in unpack(ENROLLED) if w["group"] == "operator"}
    rows = defaultdict(list)
    for r in unpack(target):
        key, world, group = r["id"].split("/")
        assert group == r["group"] and key == r["source"]
        rows[key, int(world)].append(r)
    groups = Counter()
    events = Counter()
    totals = Counter()
    with TemporaryDirectory() as tmp:
        for w in info["rows"]:
            k = w["id"]
            src = originals[k]
            assert k in eligible and w["user_literal"] in src["question"]
            assert w["user_literal"] != w["canonical_key"]
            for v, receipt in zip(src["versions"], w["checked_versions"], strict=True):
                world = v["world"]
                totals["worlds"] += 1
                assert receipt["world"] == world and receipt["gold"] == v["answer"]
                folder = Path(tmp) / k / str(world)
                db = mount(folder, v["rows"])
                path = folder / "records.csv"
                with path.open("w", encoding="utf-8", newline="") as f:
                    csv.writer(f).writerows(v["rows"])
                providers = {
                    t.name: t for t in [*csv_tools([path], keyed=True), *tools([db])]
                }
                args = {
                    "document": path.name,
                    "key_column": w["key_column"],
                    "value_column": w["value_column"],
                    "key": w["user_literal"],
                }
                keyed = providers["csv_lookup"].callback
                try:
                    keyed(**args)
                except ValueError as exc:
                    assert str(exc) == "No CSV record matches the key"
                else:
                    raise AssertionError("The submitted user spelling found a row")
                correct = keyed(**{**args, "key": w["canonical_key"]})
                sql = providers["database_query"].callback(
                    database_name=FILENAME, statement=src["statement"]
                )
                assert (
                    fingerprint(correct) == receipt["csv_result"]
                    and fingerprint(sql) == receipt["sql_result"]
                )
                assert (
                    canonical(correct["value"])
                    == canonical(v["answer"])
                    == canonical(sql["rows"][0][0])
                )
                if world not in w["enrolled_worlds"]:
                    assert (k, world) not in rows
                    continue
                examples = rows[k, world]
                assert len(examples) == 4
                for r in examples:
                    group = r["group"]
                    groups[group] += 1
                    msgs = [*r["messages"]]
                    if group.startswith("repair"):
                        msgs.append(r["target"])
                        msgs.append(
                            {
                                "role": "tool",
                                "content": compact(
                                    correct if group == "repair_csv_exact" else sql
                                ),
                            }
                        )
                    else:
                        assert r["target"]["content"] == v["answer"]
                    measures = session(msgs, providers)
                    assert measures["error"] == 1
                    events.update(measures)
            assert (
                len(
                    {
                        src_v["answer"]
                        for src_v in src["versions"]
                        if src_v["world"] in w["enrolled_worlds"]
                    }
                )
                == 2
            )
    receipt = {
        "training": digest(target),
        "source_questions": len(info["rows"]),
        "distinct_labeled_worlds": totals["worlds"],
        "replayed_training_turns": dict(sorted(groups.items())),
        "observed_events_in_enrolled_histories": dict(sorted(events.items())),
    }
    persist(ROOT / "csv-independent-replay.json", receipt)
    print(receipt)


if __name__ == "__main__":
    replay()
