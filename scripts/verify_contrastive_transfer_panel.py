"""Independently replay the held source environments before running either decoder."""

from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest
from loopqwen.qwen_tools import declarations
from loopqwen.research.contrastive_transfer import database
from loopqwen.research.table_worlds import FILENAME, canonical
from scripts.contrastive_transfer_panel import PANEL
from scripts.contrastive_world_material import OUT, TABLES
from scripts.table_policy_material import persist


def run():
    ref = {w["id"]: w for w in unpack(TABLES / "training.json.gz")}
    counts = Counter()
    raw = []
    pair_answers = {}
    service_options = []
    with TemporaryDirectory() as tmp:
        for w in unpack(PANEL)["rows"]:
            f = w["family"]
            providers = database(w, Path(tmp))
            assert declarations(providers) == w["defs"]
            if providers:
                sc, query = providers
                assert list(sc.callback(database_name=FILENAME)["tables"]) == [
                    "records"
                ]
                if f == "tables":
                    v = query.callback(
                        database_name=FILENAME, statement=ref[w["source"]]["statement"]
                    )
                    assert (
                        v["referenced_tables"] == ["records"]
                        and not v["truncated"]
                        and canonical(v["rows"][0][0]) == canonical(w["gold"])
                    )
                else:
                    v = query.callback(
                        database_name=FILENAME,
                        statement="SELECT * FROM records ORDER BY rowid",
                    )
                    assert (
                        not v["truncated"] and [v["columns"], *v["rows"]] == w["rows"]
                    )
                    service_options.append(len(v["rows"]))
            if f == "locale":
                assert fingerprint(w["initial_ids"]) == w["initial_sha256"]
                raw.append(
                    {
                        "id": w["id"],
                        "tokens": len(w["initial_ids"]),
                        "sha256": w["initial_sha256"],
                    }
                )
            if f != "service":
                pair_answers.setdefault((f, w["cluster"]), set()).add(w["gold"])
            counts[f] += 1
    assert all(len(choices) == 2 for choices in pair_answers.values())
    persist(
        OUT / "transfer-environment-replay.json",
        {
            "material": digest(PANEL),
            "cases": dict(counts),
            "different_answers_per_source": 2,
            "paired_non_service_sources": len(pair_answers),
            "complete_original_service_rows": sum(service_options),
            "minimum_service_options": min(service_options),
            "maximum_service_options": max(service_options),
            "raw_locale_inputs": raw,
        },
    )
    print(counts, sum(service_options), [w["tokens"] for w in raw], flush=True)


if __name__ == "__main__":
    run()
