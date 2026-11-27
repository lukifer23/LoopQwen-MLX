"""Mount real provider histories and compare the last successful SQL against all frozen variants."""

import json
import sqlite3
from pathlib import Path
from zipfile import ZipFile

from scripts.dialogue_database_sources import CACHE
from scripts.dialogue_database_suite import SUITE
from scripts.provider_field_mutations import OUTPUT, variant
from scripts.provider_recovery_field import PANEL

from ..io import digest, read
from ..relational_tools import select, tools
from .native_function_study import executions
from .table_worlds import canonical, present


class FieldDatabases:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.sources = {
            k: v
            for group in read(PANEL.with_name("field-denotations.json"))[
                "panels"
            ].values()
            for k, v in group["by_database"].items()
        }
        self.generated = read(OUTPUT)["databases"]
        self.paths = {}
        self.results = {}

    def mount(self, db):
        if db not in self.paths:
            source = self.sources[db]
            directory = self.folder / db
            directory.mkdir(parents=True, exist_ok=True)
            original = directory / (db + ".sqlite")
            with ZipFile(CACHE) as z:
                original.write_bytes(z.read(source["original"]["member"]))
            assert digest(original) == source["original"]["sha256"]
            paths = [(original, source["original_denotations"])]
            with ZipFile(SUITE) as suite:
                for item in source["distilled"]:
                    if item["named_original"]:
                        continue
                    path = directory / Path(item["member"]).name
                    path.write_bytes(suite.read(item["member"]))
                    assert digest(path) == item["sha256"]
                    paths.append((path, item["denotations"]))
            for item in self.generated.get(db, []):
                path = directory / f"subset-{item['index']}.sqlite"
                path.write_bytes(original.read_bytes())
                assert variant(path, db, item["index"]) == item["counts"]
                assert digest(path) == item["sha256"]
                paths.append((path, item["denotations"]))
            self.paths[db] = paths
        return self.paths[db]

    def providers(self, row):
        return tools([self.mount(row["source"])[0][0]])

    def program(self, row, statement):
        db = row["source"]
        base = row["id"].rsplit("/", 1)[0] + "/unassisted"
        counts = {"tested": 0, "agreed": 0, "errors": 0, "real_reads": 0}
        for path, expected in self.mount(db):
            key = (path, statement)
            if key not in self.results:
                try:
                    self.results[key] = select(path, statement)
                except (sqlite3.Error, ValueError, TypeError, OverflowError):
                    self.results[key] = None
            actual = self.results[key]
            counts["tested"] += 1
            if actual is None:
                counts["errors"] += 1
            elif actual["referenced_tables"] and not actual["truncated"]:
                counts["real_reads"] += 1
                counts["agreed"] += actual["rows"] == expected[base]
        counts["equivalent_on_suite"] = counts["agreed"] == counts["tested"]
        return counts

    def score(self, row, tr):
        # The initial histories and all new callbacks receive the same kind of check.
        # The last successful query is the one whose evidence the answer can claim.
        prior = []
        if row["condition"] != "unassisted":
            fn = row["messages"][-2]["tool_calls"][0]["function"]
            prior = [
                {"call": fn, "observation": json.loads(row["messages"][-1]["content"])}
            ]
        callbacks = executions(tr)
        successful = [
            (fresh, e)
            for fresh, items in ((False, prior), (True, callbacks))
            for e in items
            if e["call"]["name"] == "database_query"
            and e["observation"].get("referenced_tables")
            and not e["observation"].get("truncated", True)
        ]
        proof = None
        fresh_read = False
        if successful:
            fresh_read, event = successful[-1]
            a = event["call"]["arguments"]
            if a.get("database_name") == row["source"] + ".sqlite" and event[
                "observation"
            ]["rows"] == [[row["gold"]]]:
                proof = self.program(row, a["statement"])
        finished = tr["status"] == "answered"
        reply = tr["answer"] or ""
        delivered = finished and (
            present(reply, row["gold"]) or canonical(reply) == canonical(row["human"])
        )
        grounded = bool(
            proof
            and proof["equivalent_on_suite"]
            and (fresh_read or row["condition"] == "valid_current")
        )
        return {
            "completed": finished,
            "closed_form_delivery": delivered,
            "last_successful_is_fresh": fresh_read,
            "suite": proof,
            "grounded": grounded,
            "success": delivered and grounded,
        }
