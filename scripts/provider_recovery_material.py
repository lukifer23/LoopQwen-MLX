"""Construct failed and corrected tool turns from original human queries and live databases."""

import argparse
from collections import Counter
from pathlib import Path
from sqlite3 import OperationalError
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.relational_tools import tools
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from scripts.dialogue_database_material import PANEL, TRAIN
from scripts.dialogue_database_sources import CACHE, DEST
from scripts.program_language_material import call, norm, observed
from scripts.table_policy_material import persist

ROOT = Path("research/sources/provider-recovery")


def mount_databases(src, folder):
    databases = {}
    with ZipFile(CACHE) as z:
        for w in src["cohorts"]["train"]:
            name = w["database"]
            p = folder / (name + ".sqlite")
            p.write_bytes(z.read(w["member"]))
            assert digest(p) == w["sqlite_sha256"]
            databases[name] = tools([p])
    return databases


def qualify(rows, dbs):
    cases = {}
    for r in rows:
        key = r["id"].removesuffix("-" + r["group"])
        if r["group"] == "database_query":
            c = r["target"]["tool_calls"][0]["function"]["arguments"]
            cases[key] = {
                "id": key,
                "source": r["source"],
                "dialogue": r["dialogue"],
                "sql": c["statement"],
                "question": r["messages"][0]["content"],
                "messages": r["messages"],
                "tools": r["tools"],
                "target": r["target"],
            }
    for r in rows:
        if r["group"] == "database_answer":
            k = r["id"].removesuffix("-answer")
            w = cases[k]
            provider = dbs[w["source"]][1]
            answer = provider.callback(
                database_name=w["source"] + ".sqlite", statement=w["sql"]
            )
            assert (
                answer["referenced_tables"]
                and answer["rows"]
                and not answer["truncated"]
            )
            assert (
                observed(answer) == r["messages"][-1]
                and r["messages"][:-2] == w["messages"]
                and r["messages"][-2] == w["target"]
            )
            w.update(returned=answer, answer=r["target"])
    assert len(cases) == 220 and all("answer" in r for r in cases.values())
    return cases


def lexical(s):
    return set(norm(s).split())


def similarity(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def crossed(current, cases, dbs):
    # Prefer another author's query for a similar current utterance. Exclude the
    # current database and previously seen SQL. Both source programs run unmodified.
    current_question = lexical(current["question"].strip().splitlines()[-1])
    current_sql = lexical(current["sql"])
    candidates = sorted(
        (
            w
            for w in cases.values()
            if w["source"] != current["source"] and w["sql"] != current["sql"]
        ),
        key=lambda w: (
            -similarity(
                current_question, lexical(w["question"].strip().splitlines()[-1])
            ),
            -similarity(current_sql, lexical(w["sql"])),
            fingerprint([52092, current["id"], w["id"]]),
        ),
    )
    _, query = dbs[current["source"]]
    name = current["source"] + ".sqlite"
    for w in candidates:
        params = {"database_name": name, "statement": w["sql"]}
        try:
            query.callback(**params)
        except OperationalError as err:
            if not str(err).lower().startswith(("no such table:", "no such column:")):
                continue
            # The normal chat runtime sends the exception class to the model.
            error = observed({"error": type(err).__name__})
            action = call(query, **params)
            return w, action, error
    return None


def material():
    source = read(DEST / "originals.json")
    assert digest(CACHE) == source["archive_sha256"]
    train = read(TRAIN)["examples"]
    blocked = {r["source"] for r in read(PANEL)["rows"]}
    assert not {r["source"] for r in train} & blocked
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    out = []
    origins = []
    excluded = Counter()
    with TemporaryDirectory() as t:
        dbs = mount_databases(source, Path(t))
        cases = qualify(train, dbs)
        for w in cases.values():
            hit = crossed(w, cases, dbs)
            if hit is None:
                excluded["no_crossed_provider_error"] += 1
                continue
            other, wrong, error = hit
            prompt, *schema_history = w["messages"]
            assert len(schema_history) == 2 and "tool_calls" in schema_history[0]
            prior = [prompt, wrong, error]
            if w["sql"] in prompt["content"]:
                excluded["expert_statement_already_in_request"] += 1
                continue
            corrected = (*prior, *schema_history, w["target"], observed(w["returned"]))
            examples = []
            # Both possible orders are causal: when the schema is already known,
            # emit the corrected query directly; otherwise inspect the file first.
            for kind, hist, target in (
                ("inspect_after_error", prior, schema_history[0]),
                ("repair_after_schema", [*prior, *schema_history], w["target"]),
                ("repair_known_schema", [*w["messages"], wrong, error], w["target"]),
                ("deliver_corrected_result", list(corrected), w["answer"]),
            ):
                r = {
                    "id": w["id"] + "/" + kind,
                    "group": kind,
                    "source": w["source"],
                    "dialogue": w["dialogue"],
                    "messages": hist,
                    "tools": w["tools"],
                    "target": target,
                }
                p, y = token_pair(tok, r)
                if len(p) + len(y) > 1792 or len(y) > 384:
                    break
                examples.append((r, len(p), len(y)))
            if len(examples) != 4:
                excluded["whole_source_over_length"] += 1
                continue
            out.extend(r for r, _, _ in examples)
            origins.append(
                {
                    "id": w["id"],
                    "database": w["source"],
                    "dialogue": w["dialogue"],
                    "wrong_source": other["id"],
                    "wrong_database": other["source"],
                    "original_sql": w["sql"],
                    "crossed_sql": other["sql"],
                    "exception": "OperationalError",
                    "returned": fingerprint(w["returned"]),
                    "prefix_tokens": sum(n for _, n, _ in examples),
                    "labels": sum(n for _, _, n in examples),
                }
            )
    assert len({r["id"] for r in out}) == len(out) and len(origins) + sum(
        excluded.values()
    ) == len(cases)
    return out, origins, dict(excluded)


def run(verify=False):
    rows, origins, excluded = material()
    dest = ROOT / "training.json.gz"
    ROOT.mkdir(parents=True, exist_ok=True)
    if verify or dest.exists():
        assert unpack(dest) == rows
    else:
        packed(dest, rows)
    details = {
        "source_archive": digest(CACHE),
        "input_training": digest(TRAIN),
        "prior_held": digest(PANEL),
        "training": digest(dest),
        "cases": len(origins),
        "databases": len({r["database"] for r in origins}),
        "distinct_crossed_programs": len({r["crossed_sql"] for r in origins}),
        "turns": len(rows),
        "labels": sum(r["labels"] for r in origins),
        "prefix_tokens": sum(r["prefix_tokens"] for r in origins),
        "exclusions": excluded,
        "origins": origins,
    }
    if verify:
        assert read(ROOT / "qualification.json") == details
    else:
        persist(ROOT / "qualification.json", details)
    print(
        {
            k: v
            for k, v in details.items()
            if k
            not in (
                "origins",
                "source_archive",
                "input_training",
                "prior_held",
                "training",
            )
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    run(parser.parse_args().verify)
