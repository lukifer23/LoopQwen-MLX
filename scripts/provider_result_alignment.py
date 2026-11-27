"""Pair two human questions about one table to teach when a successful SQL result answers the user."""

import csv
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen import qwen_tools
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.relational_tools import tools
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from loopqwen.research.table_worlds import (
    FILENAME,
    canonical,
    locations,
    prepare_tables,
    present,
    prompt,
    query_results,
)
from scripts.program_language_material import answer, call, norm, observed, user
from scripts.provider_recovery_csv import ENROLLED, MULTI
from scripts.provider_recovery_material import ROOT
from scripts.table_policy_material import persist

SEED = 163503


def historical():
    from ast import literal_eval

    p = Path("paper")
    with (p / "conditional-primary-requests.csv").open() as f:
        records = {r["id"]: r for r in csv.DictReader(f)}
    with (p / "conditional-primary-changed-statements.csv").open() as f:
        calls = list(csv.DictReader(f))
    report = {}
    for arm in ("candidate", "control"):
        c = Counter()
        wrong = defaultdict(list)
        for r in calls:
            if r["arm"] != arm:
                continue
            c["query_calls"] += 1
            o = literal_eval(r["observed"])
            rs = o.get("rows", [])
            if (
                r["current_valid_read"] != "True"
                or len(rs) != 1
                or len(rs[0]) != 1
                or rs[0][0] is None
            ):
                continue
            good = canonical(rs[0][0]) == canonical(r["human_answer"])
            assert good == (r["current_correct_scalar"] == "True")
            c["correct_scalars" if good else "wrong_scalars"] += 1
            if not good:
                wrong[r["id"]].append(rs[0][0])
        c["requests_with_wrong_scalar"] = len(wrong)
        c["requests_repeating_wrong_scalar_as_final"] = sum(
            any(present(records[k][arm + "_visible"], value) for value in values)
            for k, values in wrong.items()
        )
        c["requests_recovered_to_grounded_final"] = sum(
            records[k][arm + "_grounded"] == "True" for k in wrong
        )
        report[arm] = dict(c)
    return report


def number(value):
    try:
        float(str(value).replace(",", ""))
        return True
    except ValueError:
        return False


def bind(row, other, folder):
    rows = []
    facts = []
    own = query_results(row["statement"], row, folder)
    foreign = query_results(other["statement"], row, folder)
    paths = locations(folder, row)
    for v, p, yes, no in zip(row["versions"], paths, own, foreign, strict=True):
        assert yes["correct"] and len(v["votes"]) >= 3
        before = no.get("observation", {}).get("rows", [])
        if (
            not no["used"]
            or len(before) != 1
            or len(before[0]) != 1
            or before[0][0] is None
            or not canonical(before[0][0])
            or no["correct"]
            or number(before[0][0]) != number(v["answer"])
        ):
            return None
        if v["world"] == 0:
            assert canonical(before[0][0]) == canonical(other["versions"][0]["answer"])
        facts.append(
            {
                "world": v["world"],
                "correct": v["answer"],
                "different": before[0][0],
                "own_hash": fingerprint(yes["observation"]),
                "other_hash": fingerprint(no["observation"]),
            }
        )
    selected = [
        0,
        min(
            (
                i
                for i, v in enumerate(row["versions"])
                if canonical(v["answer"]) != canonical(facts[0]["correct"])
            ),
            key=lambda i: fingerprint([SEED, row["id"], facts[i]["world"]]),
        ),
    ]
    for i in selected:
        v, yes, no = row["versions"][i], own[i], foreign[i]
        schema, query = tools([paths[i]])
        schema_args = {"database_name": FILENAME}
        history = [
            user(prompt(row)),
            call(schema, **schema_args),
            observed(schema.callback(**schema_args)),
        ]
        a, b = (
            call(query, database_name=FILENAME, statement=x["statement"])
            for x in (row, other)
        )
        good = [a, observed(yes["observation"])]
        bad = [b, observed(no["observation"])]
        final = answer(v["answer"])
        for group, after, target in (
            ("answer_valid_read", good, final),
            ("redirect_valid_read", bad, a),
            ("answer_after_redirect", bad + good, final),
        ):
            rows.append(
                {
                    "id": f"{row['id']}/{v['world']}/{group}",
                    "source": row["id"],
                    "other": other["id"],
                    "world": v["world"],
                    "group": group,
                    "messages": history + after,
                    "tools": qwen_tools.declarations([schema, query]),
                    "target": target,
                }
            )
    return rows, facts


def material():
    eligible = {r["pair"] for r in unpack(ENROLLED) if r["group"] == "operator"}
    held = {
        w["context"] for w in unpack(ENROLLED.with_name("withheld.json.gz"))["tables"]
    }
    held.update(
        w["context"]
        for w in read("research/sources/changed-tables/originals.json")["cohorts"][
            "dev"
        ]
    )
    tables = defaultdict(list)
    for w in unpack(MULTI):
        if w["id"] in eligible:
            assert w["context"] not in held
            tables[w["context"]].append(w)
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    rows = []
    origins = []
    reasons = Counter()
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for ctx, options in sorted(tables.items()):
            if len(options) < 2:
                reasons["one enrolled question"] += 1
                continue
            assert len({fingerprint(w["versions"][0]["rows"]) for w in options}) == 1
            prepare_tables(folder, options)
            pairs = sorted(
                combinations(options, 2),
                key=lambda pair: fingerprint([SEED, *sorted(w["id"] for w in pair)]),
            )
            for a, b in pairs:
                if (
                    norm(a["question"]) == norm(b["question"])
                    or a["statement"] == b["statement"]
                ):
                    continue
                left = bind(a, b, folder)
                if left is None:
                    continue
                right = bind(b, a, folder)
                if right is None:
                    continue
                new = left[0] + right[0]
                sizes = [tuple(map(len, token_pair(tok, r))) for r in new]
                if any(p + y > 1792 or y > 384 for p, y in sizes):
                    continue
                for x, y in ((left[0], right[0]), (right[0], left[0])):
                    # For the original table, the same genuine query and returned
                    # observation require opposite actions solely because the user changes.
                    assert x[0]["messages"][1:] == y[1]["messages"][1:]
                    assert (
                        x[0]["tools"] == y[1]["tools"]
                        and x[0]["messages"][0] != y[1]["messages"][0]
                    )
                rows.extend(new)
                origins.append(
                    {
                        "context": ctx,
                        "sources": [a["id"], b["id"]],
                        "questions": [a["question"], b["question"]],
                        "programs": [a["statement"], b["statement"]],
                        "checked": [left[1], right[1]],
                        "prefix_tokens": sum(x for x, y in sizes),
                        "labels": sum(y for x, y in sizes),
                    }
                )
                break
            else:
                reasons[
                    "no bidirectional scalar contrast on all versions within limits"
                ] += 1
    return rows, origins, dict(reasons)


def run():
    rows, origins, excluded = material()
    assert len(rows) == len(origins) * 12 and len({r["id"] for r in rows}) == len(rows)
    dest = ROOT / "semantic-training.json.gz"
    if dest.exists():
        assert unpack(dest) == rows
    else:
        packed(dest, rows)
    info = {
        "training_sha256": digest(dest),
        "source": digest(MULTI),
        "eligible": digest(ENROLLED),
        "historical_wrong_scalar": historical(),
        "tables": len(origins),
        "questions": len(origins) * 2,
        "checked_question_versions": sum(len(v) for o in origins for v in o["checked"]),
        "prefix_tokens": sum(o["prefix_tokens"] for o in origins),
        "labels": sum(o["labels"] for o in origins),
        "supervised_turns": len(rows),
        "groups": dict(Counter(r["group"] for r in rows)),
        "excluded": excluded,
        "origins": origins,
    }
    persist(ROOT / "semantic-qualification.json", info)
    print({k: v for k, v in info.items() if k != "origins"})


if __name__ == "__main__":
    run()
