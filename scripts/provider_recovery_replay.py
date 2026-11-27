"""Rebuild source DBs and replay every enrolled provider action and target from its JSON."""

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.relational_tools import tools
from loopqwen.tool_session import compact
from scripts.dialogue_database_material import TRAIN
from scripts.dialogue_database_sources import CACHE, DEST
from scripts.provider_recovery_material import ROOT
from scripts.table_policy_material import persist


def session(messages, providers):
    actions = Counter()
    observed = None
    for msg in messages:
        if msg["role"] == "tool":
            assert observed is not None and compact(observed) == msg["content"]
            observed = None
            continue
        assert observed is None
        if "tool_calls" not in msg:
            continue
        (event,) = msg["tool_calls"]
        f = event["function"]
        try:
            observed = providers[f["name"]].callback(**f["arguments"])
        except sqlite3.OperationalError:
            observed = {"error": "OperationalError"}
        actions["error" if "error" in observed else f["name"]] += 1
    assert observed is None
    return actions


def verify():
    q = read(ROOT / "qualification.json")
    assert (
        digest(ROOT / "training.json.gz") == q["training"]
        and digest(CACHE) == q["source_archive"]
    )
    sources = {
        r["database"]: r for r in read(DEST / "originals.json")["cohorts"]["train"]
    }
    original = {r["id"]: r for r in read(TRAIN)["examples"]}
    by_case = defaultdict(dict)
    for row in unpack(ROOT / "training.json.gz"):
        key = row["id"].rsplit("/", 1)[0]
        assert row["group"] not in by_case[key]
        by_case[key][row["group"]] = row
    counts = Counter()
    with TemporaryDirectory() as tmp, ZipFile(CACHE) as packed:
        providers = {}
        for key, w in sources.items():
            dest = Path(tmp) / (key + ".sqlite")
            dest.write_bytes(packed.read(w["member"]))
            assert digest(dest) == w["sqlite_sha256"]
            providers[key] = {t.name: t for t in tools([dest])}
        for w in q["origins"]:
            key = w["id"]
            specs = by_case[key]
            current = providers[w["database"]]
            foreign = providers[w["wrong_database"]]
            # Independently require that the proposed action succeeds in its origin,
            # raises on the actual source, and the human correction returns the original cells.
            assert w["database"] != w["wrong_database"]
            wrong = foreign["database_query"].callback(
                database_name=w["wrong_database"] + ".sqlite",
                statement=w["crossed_sql"],
            )
            assert wrong["rows"] and wrong["referenced_tables"]
            target = current["database_query"].callback(
                database_name=w["database"] + ".sqlite", statement=w["original_sql"]
            )
            assert fingerprint(target) == w["returned"]
            assert (
                specs["deliver_corrected_result"]["target"]
                == original[key + "-answer"]["target"]
            )
            assert (
                specs["repair_after_schema"]["target"]
                == specs["repair_known_schema"]["target"]
                == original[key + "-database_query"]["target"]
            )
            assert (
                specs["inspect_after_error"]["target"]
                == original[key + "-database_schema"]["target"]
            )
            for r in specs.values():
                messages = [*r["messages"]]
                if "tool_calls" in r["target"]:
                    messages.append(r["target"])
                    expected = (
                        target
                        if r["group"].startswith("repair")
                        else json.loads(
                            original[key + "-database_query"]["messages"][-1]["content"]
                        )
                    )
                    messages.append({"role": "tool", "content": compact(expected)})
                result = session(messages, current)
                assert result["error"] == 1
                counts.update(result)
    persist(
        ROOT / "independent-replay.json",
        {
            "training": q["training"],
            "sources": len(by_case),
            "turns": sum(len(v) for v in by_case.values()),
            "action_outcomes": dict(sorted(counts.items())),
            "original_and_changed_calls_verified": len(q["origins"]),
            "original_human_replies_verified": len(q["origins"]),
        },
    )
    print(dict(counts))


if __name__ == "__main__":
    verify()
