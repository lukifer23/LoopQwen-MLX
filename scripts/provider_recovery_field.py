"""Reserve complete original requests on unused development databases and physically execute both histories."""

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from loopqwen import qwen_tools
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.relational_tools import tools
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.table_worlds import canonical
from scripts.dialogue_database_sources import CACHE, DEST, SHA
from scripts.program_language_material import call, norm, observed, user
from scripts.provider_recovery_material import ROOT
from scripts.provider_result_alignment import number
from scripts.table_policy_material import persist

SEED = 1127131
PANEL = ROOT / "development-requests.json.gz"


def released(z, dbs, split):
    pool = defaultdict(list)
    official = json.loads(
        z.read(f"cosql_dataset/sql_state_tracking/cosql_{split}.json")
    )

    def sql_key(text):
        return "".join(text.casefold().split()).rstrip(";")

    published = {
        (r["database_id"], norm(t["utterance"]), sql_key(t["query"]))
        for r in official
        for t in r["interaction"]
    }
    for key, d in json.loads(
        z.read("cosql_dataset/cosql_all_info_dialogs.json")
    ).items():
        db = d["db_id"]
        if db not in dbs or d.get("disabled"):
            continue
        turns = d["turns"]
        first = next((i for i, t in enumerate(turns) if t.get("isSql")), 0)
        if not 0 < first < len(turns) - 1:
            continue
        question, sql, reply = turns[first - 1 : first + 2]
        if not (
            question.get("isUser")
            and "INFORM_SQL" in (question.get("label") or [])
            and "CONFIRM_SQL" in (reply.get("label") or [])
        ):
            continue
        # A first substantive turn stands alone. Retain no later dialogue annotation
        # in the prompt, and exclude questions whose earlier user content is needed.
        if any(
            t.get("isUser") and set(t.get("label") or []) - {"GREETING", "WELCOME"}
            for t in turns[: first - 1]
        ):
            continue
        if (db, norm(question["text"]), sql_key(sql["rawSql"])) not in published:
            continue
        result = sql.get("sql_result")
        if (
            not result
            or len(result) != 2
            or len(result[1]) != 1
            or result[1][0] in (None, "")
        ):
            continue
        value = result[1][0]
        if not re.search(
            r"(?<!\w)" + re.escape(str(value)) + r"(?!\w)",
            reply["text"],
            flags=re.IGNORECASE,
        ):
            continue
        pool[db].append(
            {
                "dialogue": key,
                "source_turn": first - 1,
                "question": question["text"],
                "sql": sql["rawSql"].strip(),
                "human": reply["text"],
                "gold": value,
            }
        )
    return pool


def build(split="dev"):
    assert digest(CACHE) == SHA
    prior = read(DEST / "originals.json")["cohorts"]
    excluded = {r["database"] for side in prior.values() for r in side}
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    sources = {}
    cases = []
    eligible = {}
    with ZipFile(CACHE) as z, TemporaryDirectory() as tmp:
        dev = {
            r["database_id"]
            for r in json.loads(
                z.read(f"cosql_dataset/sql_state_tracking/cosql_{split}.json")
            )
        } - excluded
        pool = released(z, dev, split)
        if split == "dev":
            assert len(dev) == 8
        providers = {}
        selected = {}
        for db in sorted(
            dev,
            key=(
                None
                if split == "dev"
                else lambda k: fingerprint([SEED, "unseen schema", k])
            ),
        ):
            if split == "train" and len(selected) == 7:
                break
            path = Path(tmp) / (db + ".sqlite")
            member = f"cosql_dataset/database/{db}/{db}.sqlite"
            path.write_bytes(z.read(member))
            schema, reader = providers[db] = tools([path])
            sources[db] = {
                "member": member,
                "sha256": digest(path),
                "bytes": path.stat().st_size,
            }
            verified = []
            for w in pool[db]:
                try:
                    r = reader.callback(database_name=path.name, statement=w["sql"])
                except (ValueError, sqlite3.Error):
                    continue
                if (
                    r["rows"] == [[w["gold"]]]
                    and r["referenced_tables"]
                    and not r["truncated"]
                ):
                    verified.append({**w, "returned": r})
            pairs = [
                (a, b)
                for a, b in combinations(verified, 2)
                if a["dialogue"] != b["dialogue"]
                and norm(a["question"]) != norm(b["question"])
                and a["sql"] != b["sql"]
                and canonical(a["gold"]) != canonical(b["gold"])
                and number(a["gold"]) == number(b["gold"])
            ]
            eligible[db] = {
                "first_human_scalar_questions": len(verified),
                "same_type_distinct_pairs": len(pairs),
            }
            if not pairs:
                continue
            selected[db] = min(
                pairs,
                key=lambda p: (
                    not bool(
                        set(p[0]["returned"]["referenced_tables"])
                        & set(p[1]["returned"]["referenced_tables"])
                    ),
                    fingerprint([SEED, db, sorted(w["dialogue"] for w in p)]),
                ),
            )
        for db, pair in sorted(selected.items()):
            schema, reader = providers[db]
            args = {"database_name": db + ".sqlite"}
            definitions = qwen_tools.declarations([schema, reader])
            header = [call(schema, **args), observed(schema.callback(**args))]
            for i, current in enumerate(pair):
                other = pair[1 - i]
                key = f"cosql-{split}/{db}/{current['dialogue']}"
                msg = [user(current["question"])]
                candidates = sorted(
                    ((k, r) for k, rs in selected.items() if k != db for r in rs),
                    key=lambda x: fingerprint([SEED, key, x[0], x[1]["dialogue"]]),
                )
                for foreign_db, foreign in candidates:
                    try:
                        reader.callback(**args, statement=foreign["sql"])
                    except sqlite3.OperationalError as e:
                        if str(e).startswith(("no such table:", "no such column:")):
                            break
                else:
                    raise ValueError("Every crossed query succeeded")
                for condition, previous, result in (
                    ("unassisted", None, None),
                    ("valid_current", current, current["returned"]),
                    ("valid_other", other, other["returned"]),
                    ("provider_error", foreign, {"error": "OperationalError"}),
                ):
                    history = (
                        msg
                        if previous is None
                        else [
                            *msg,
                            *header,
                            call(reader, **args, statement=previous["sql"]),
                            observed(result),
                        ]
                    )
                    ids = tok.encode(
                        qwen_tools.render(tok, history, definitions),
                        add_special_tokens=False,
                    )
                    assert len(ids) <= 2500
                    cases.append(
                        {
                            "id": key + "/" + condition,
                            "source": db,
                            "condition": condition,
                            "dialogue": current["dialogue"],
                            "source_turn": current["source_turn"],
                            "question": current["question"],
                            "gold": current["gold"],
                            "human": current["human"],
                            "expert_sql": current["sql"],
                            "expert_result": current["returned"],
                            "other_dialogue": previous["dialogue"]
                            if previous
                            else None,
                            "query_origin_database": foreign_db
                            if condition == "provider_error"
                            else db
                            if previous
                            else None,
                            "messages": history,
                            "tools": definitions,
                            "initial_ids": ids,
                            "prompt_sha256": fingerprint(ids),
                        }
                    )
        for db, ww in selected.items():
            for a, b in (ww, list(reversed(ww))):
                same = next(
                    r
                    for r in cases
                    if r["dialogue"] == a["dialogue"]
                    and r["condition"] == "valid_current"
                )
                changed = next(
                    r
                    for r in cases
                    if r["dialogue"] == b["dialogue"]
                    and r["condition"] == "valid_other"
                )
                assert (
                    same["messages"][1:] == changed["messages"][1:]
                    and same["messages"][0] != changed["messages"][0]
                )
    order = sorted(cases, key=lambda r: fingerprint([SEED, r["id"]]))
    return {"rows": order, "databases": sources, "upstream": SHA, "eligible": eligible}


def run(verify=False, split="dev"):
    result = build(split)
    path = (
        PANEL if split == "dev" else ROOT / "withheld-training-split-requests.json.gz"
    )
    if verify or path.exists():
        assert result == unpack(path)
    else:
        packed(path, result)
    rows = result["rows"]
    meta = {
        "source": SHA,
        "panel": digest(path),
        "databases": result["eligible"],
        "distinct_questions": len({r["dialogue"] for r in rows}),
        "mounted_databases": len({r["source"] for r in rows}),
        "counts": dict(Counter(r["condition"] for r in rows)),
        "total_opening_tokens": sum(len(r["initial_ids"]) for r in rows),
        "source_exclusion": "Official CoSQL development databases absent from all databases in the project's earlier original CoSQL cohort. Only the original first independently executable human SQL question from a dialogue, with a verified nonempty scalar explicitly stated in the human response, is eligible. Two same-type different-answer questions per enrolled database. Prior successful or failing calls in three conditions are controlled interventions. No model outcomes are used in enrollment.",
    }
    if split == "train":
        assert meta["mounted_databases"] == 7
        meta["source_exclusion"] = meta["source_exclusion"].replace(
            "Official CoSQL development databases",
            "Seven hash-selected databases from the official CoSQL training split",
        )
    persist(
        ROOT
        / (
            "development-inventory.json"
            if split == "dev"
            else "withheld-training-split-inventory.json"
        ),
        meta,
    )
    print(meta)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    p.add_argument("--split", choices=("dev", "train"), default="dev")
    args = p.parse_args()
    run(args.verify, args.split)
