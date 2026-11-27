"""Recover exact CSV keys using intact user wording and real reads of the same source."""

import csv
import re
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen import qwen_tools
from loopqwen.csv_tools import tools as csv_tools
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest
from loopqwen.relational_tools import tools as db_tools
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from loopqwen.research.table_worlds import FILENAME, canonical, mount
from scripts.program_language_material import answer, call, observed, user
from scripts.provider_recovery_material import ROOT
from scripts.table_policy_material import persist

MULTI = Path("research/sources/multiworld-programs/training.json.gz")
ENROLLED = Path("research/sources/contrastive-worlds/training.json.gz")


def identify(w, *, case_change=True):
    versions = w["versions"]
    header, *body = versions[0]["rows"]
    choices = set()
    for row in body:
        for i, key in enumerate(row):
            if not key.strip():
                continue
            hit = re.search(
                r"(?<!\w)" + re.escape(key) + r"(?!\w)",
                w["question"],
                flags=re.IGNORECASE,
            )
            if not hit or (case_change and hit[0] == key):
                continue
            for j, v in enumerate(row):
                if i == j or canonical(v) != canonical(versions[0]["answer"]):
                    continue
                for world in versions:
                    matches = [r for r in world["rows"][1:] if r[i] == key]
                    if len(matches) != 1 or canonical(matches[0][j]) != canonical(
                        world["answer"]
                    ):
                        break
                else:
                    choices.add((header[i], key, hit[0], header[j]))
    return sorted(choices)


def bind(folder, w, world, link):
    key_column, key, literal, value_column = link
    folder.mkdir(exist_ok=True, parents=True)
    p = folder / "records.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(world["rows"])
    (keyed,) = csv_tools([p], keyed=True)
    schema, sql = db_tools([mount(folder, world["rows"])])
    args = {
        "document": p.name,
        "key_column": key_column,
        "key": key,
        "value_column": value_column,
    }
    bad = {**args, "key": literal}
    # A failure must be genuine even when the literal or a similarly spelled row exists.
    try:
        keyed.callback(**bad)
    except ValueError as e:
        if str(e) != "No CSV record matches the key":
            return None
    else:
        return None
    result = keyed.callback(**args)
    sql_args = {"database_name": FILENAME, "statement": w["statement"]}
    executed = sql.callback(**sql_args)
    if not (
        canonical(result["value"]) == canonical(world["answer"])
        and executed["referenced_tables"] == ["records"]
        and not executed["truncated"]
        and len(executed["rows"]) == 1
        and len(executed["rows"][0]) == 1
        and canonical(executed["rows"][0][0]) == canonical(world["answer"])
    ):
        return None
    col = '"' + key_column.replace('"', '""') + '"'
    quoted = "'" + literal.replace("'", "''") + "'"
    discovery = {
        "database_name": FILENAME,
        "statement": "SELECT DISTINCT "
        + col
        + " FROM records WHERE "
        + col
        + " = "
        + quoted
        + " COLLATE NOCASE",
    }
    lookup = sql.callback(**discovery)
    if lookup["rows"] != [[key]]:
        return None
    schema_args = {"database_name": FILENAME}
    opening = [
        user(
            "Use the supplied CSV or SQLite copy of the same table to answer the question.\n"
            + w["question"]
        )
    ]
    schema_history = [
        *opening,
        call(schema, **schema_args),
        observed(schema.callback(**schema_args)),
    ]
    failed = [call(keyed, **bad), observed({"error": "ValueError"})]
    knows_key = [*schema_history, call(sql, **discovery), observed(lookup), *failed]
    knows_schema = [*schema_history, *failed]
    via_csv, via_sql = call(keyed, **args), call(sql, **sql_args)
    target = answer(world["answer"])
    turns = []
    for k, hist, y in (
        ("repair_csv_exact", knows_key, via_csv),
        ("repair_via_sql", knows_schema, via_sql),
        ("deliver_csv", [*knows_key, via_csv, observed(result)], target),
        ("deliver_sql", [*knows_schema, via_sql, observed(executed)], target),
    ):
        turns.append(
            {
                "id": w["id"] + "/" + str(world["world"]) + "/" + k,
                "group": k,
                "source": w["id"],
                "database": w["context"],
                "messages": hist,
                "tools": qwen_tools.declarations([keyed, schema, sql]),
                "target": y,
            }
        )
    return turns, {
        "world": world["world"],
        "gold": world["answer"],
        "key": fingerprint(lookup),
        "csv_result": fingerprint(result),
        "sql_result": fingerprint(executed),
    }


def build():
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    ids = {w["pair"] for w in unpack(ENROLLED) if w["group"] == "operator"}
    assert len(ids) == 942
    out = []
    meta = []
    excluded = Counter()
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for w in unpack(MULTI):
            if w["id"] not in ids:
                continue
            links = identify(w)
            if len(links) != 1:
                excluded["case_mismatched_unambiguous_lookup_absent"] += 1
                continue
            worlds = w["versions"]
            other = min(
                (
                    v
                    for v in worlds[1:]
                    if canonical(v["answer"]) != canonical(worlds[0]["answer"])
                ),
                key=lambda v: fingerprint([17235, w["id"], v["world"]]),
            )
            keep = {worlds[0]["world"], other["world"]}
            seq = []
            checked = []
            counts = Counter()
            for v in worlds:
                val = bind(folder / w["id"] / str(v["world"]), w, v, links[0])
                if val is None:
                    break
                rows, receipt = val
                checked.append(receipt)
                if v["world"] in keep:
                    for r in rows:
                        p, y = token_pair(tok, r)
                        if len(p) + len(y) > 1792 or len(y) > 384:
                            break
                        counts["prefix_tokens"] += len(p)
                        counts["labels"] += len(y)
                        seq.append(r)
            if len(checked) != len(worlds):
                excluded["live_case_or_lookup_failed_on_any_world"] += 1
            elif len(seq) != 8:
                excluded["complete_pair_over_length"] += 1
            else:
                out.extend(seq)
                meta.append(
                    {
                        "id": w["id"],
                        "context": w["context"],
                        "question": w["question"],
                        "key_column": links[0][0],
                        "canonical_key": links[0][1],
                        "user_literal": links[0][2],
                        "value_column": links[0][3],
                        "checked_versions": checked,
                        "enrolled_worlds": sorted(keep),
                        **counts,
                    }
                )
    assert len(meta) + sum(excluded.values()) == len(ids) and len(out) == 8 * len(meta)
    target = ROOT / "csv-training.json.gz"
    if target.is_file():
        assert unpack(target) == out
    else:
        packed(target, out)
    obj = {
        "qualified": len(meta),
        "turns": len(out),
        "worlds_replayed": sum(len(w["checked_versions"]) for w in meta),
        "prefix_tokens": sum(w["prefix_tokens"] for w in meta),
        "labels": sum(w["labels"] for w in meta),
        "excluded": excluded,
        "training": digest(target),
        "original_tables": digest(MULTI),
        "eligible_training_cohort": digest(ENROLLED),
        "rows": meta,
    }
    persist(ROOT / "csv-qualification.json", obj)
    print(
        {
            k: v
            for k, v in obj.items()
            if k
            not in ("rows", "training", "original_tables", "eligible_training_cohort")
        },
        flush=True,
    )


if __name__ == "__main__":
    build()
