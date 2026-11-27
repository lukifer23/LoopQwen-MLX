"""Qualify human database requests for training on the recurrent model's own tool history."""

import sqlite3
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.research.table_worlds import canonical
from scripts.dialogue_database_sources import CACHE, DEST, SHA
from scripts.program_language_material import norm
from scripts.provider_field_mutations import LOCAL
from scripts.provider_recovery_field import PANEL, released
from scripts.provider_recovery_material import mount_databases
from scripts.provider_result_alignment import number
from scripts.table_policy_material import persist

OUT = Path("research/sources/native-tool-histories/training.json")
SEED = 769453


def collect():
    original = read(DEST / "originals.json")
    assert digest(CACHE) == original["archive_sha256"] == SHA
    records = []
    counts = Counter()
    with TemporaryDirectory() as folder, ZipFile(CACHE) as z:
        databases = mount_databases(original, Path(folder))
        assert set(databases).isdisjoint(
            w["database"] for w in original["cohorts"]["dev"]
        )
        for held in (PANEL, LOCAL):
            assert set(databases).isdisjoint(w["source"] for w in unpack(held)["rows"])
        source = released(z, databases, "train")
        for db, (_, tool) in sorted(databases.items()):
            name = db + ".sqlite"
            unique = {}
            for w in sorted(
                source[db], key=lambda w: fingerprint([SEED, w["dialogue"]])
            ):
                try:
                    receipt = tool.callback(database_name=name, statement=w["sql"])
                except (ValueError, sqlite3.Error):
                    continue
                if (
                    receipt["rows"] == [[w["gold"]]]
                    and receipt["referenced_tables"]
                    and not receipt["truncated"]
                ):
                    unique.setdefault(norm(w["question"]), dict(**w, returned=receipt))
            counts["physically_verified_unique_requests"] += len(unique)
            unused = list(unique.values())
            pairs = 0
            while unused and pairs < 6:
                left = unused.pop(0)
                possible = [
                    right
                    for right in unused
                    if number(left["gold"]) == number(right["gold"])
                    and canonical(left["gold"]) != canonical(right["gold"])
                    and left["dialogue"] != right["dialogue"]
                    and left["sql"] != right["sql"]
                ]
                if not possible:
                    continue
                right = min(
                    possible,
                    key=lambda w: (
                        not bool(
                            set(left["returned"]["referenced_tables"])
                            & set(w["returned"]["referenced_tables"])
                        ),
                        fingerprint([SEED, left["dialogue"], w["dialogue"]]),
                    ),
                )
                unused.remove(right)
                records.append(
                    dict(
                        database=db,
                        member_sha256=digest(Path(folder) / name),
                        requests=[left, right],
                    )
                )
                pairs += 1
    ids = [r["dialogue"] for pair in records for r in pair["requests"]]
    assert len(ids) == len(set(ids))
    counts.update(
        pairs=len(records),
        human_requests=len(ids),
        schemas=len({r["database"] for r in records}),
    )
    return dict(
        archive_sha256=SHA,
        earlier_training_cohort=digest(DEST / "originals.json"),
        seed=SEED,
        counts=dict(counts),
        pairs=records,
        selection="Only named original training schemas, original independently executable human first requests with expert confirmation and matching released SQL. Distinct same-type scalar answers on one schema per pair. Disjoint dialogues, capped at six pairs per database in fixed order. Every expert statement is executed on the original file through the public read-only provider. No model samples or test outcomes enter selection.",
    )


if __name__ == "__main__":
    material = collect()
    persist(OUT, material)
    print(material["counts"])
