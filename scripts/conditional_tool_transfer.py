"""Replay every freely emitted changed-table query and recover the exact lower-cap accuracy curves."""

from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from loopqwen.research.conditional_primary import ARMS, FRESH
from loopqwen.research.native_function_study import executions
from loopqwen.research.table_worlds import (
    SOURCE,
    canonical,
    prepare_tables,
    query_results,
)
from scripts.assistance_deliberation import write_csv
from scripts.conditional_reading import MASK, RATINGS, ROOT
from scripts.table_policy_reading import committed

P = Path("paper")


def build():
    for file in (
        MASK,
        RATINGS,
        Path("docs/measurements/conditional-reasoning-blinded-ratings.json"),
    ):
        committed(file)
    mask = read(MASK)
    grades = read(RATINGS)
    panel = read(read(ROOT / "registration.json")["material"])["rows"]
    tables = {w["id"]: w for w in read(SOURCE)["cohorts"]["dev"]}
    summary = {}
    details = []
    curves = []
    pair_ledger = []
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        prepare_tables(folder, tables.values())
        for a in ARMS:
            tally = Counter()
            literals = Counter()
            by_pair = defaultdict(list)
            cap_totals = {cap: Counter() for cap in (32, 64, 128, 256, 384)}
            for w, record in zip(panel, unpack(ROOT / (a + ".json.gz")), strict=True):
                assert record["id"] == w["id"]
                (q,) = record["requests"]
                tr = q["trajectory"]
                text = tr["answer"] or ""
                f = w["family"]
                full = 0
                if w["id"] in mask:
                    k = w["id"]
                    i = mask[k]["replies"].index(text)
                    full = int(
                        grades[k]["replies"][i]["rating"] == "fully usable"
                        and (f != "natural_mounted" or q["score"]["supported_read"])
                    )
                maxlen = max(len(t["token_ids"]) for t in tr["turns"])
                for cap, tt in cap_totals.items():
                    survives = maxlen <= cap
                    tt["completion"] += survives and q["score"]["completed"]
                    tt["objective"] += bool(
                        f not in FRESH
                        and f != "open"
                        and survives
                        and q["score"]["success"]
                    )
                    tt["masked_grounded"] += survives * full
                if f != "changed_table":
                    continue
                table = tables[w["row_id"]]
                i = next(
                    i
                    for i, v in enumerate(table["versions"])
                    if v["world"] == w["world"]
                )
                used = both = scalar = False
                for ordinal, e in enumerate(executions(tr), 1):
                    if e["call"]["name"] != "database_query":
                        continue
                    statement = e["call"]["arguments"].get("statement")
                    if not isinstance(statement, str):
                        continue
                    result = query_results(statement, table, folder)
                    curr, opposite = result[i], result[1 - i]
                    if "observation" in curr:
                        assert curr["observation"] == e["observation"]
                    used |= curr["used"]
                    scalar |= curr["correct"]
                    both |= curr["correct"] and opposite["correct"]
                    details.append(
                        {
                            "id": w["id"],
                            "cluster": w["cluster"],
                            "arm": a,
                            "ordinal": ordinal,
                            "world": w["world"],
                            "human_answer": w["gold"],
                            "sql": statement,
                            "observed": e["observation"],
                            "current_valid_read": curr["used"],
                            "current_correct_scalar": curr["correct"],
                            "same_statement_opposite_correct_scalar": opposite[
                                "correct"
                            ],
                        }
                    )
                tally["cases"] += 1
                tally["any_tool"] += bool(executions(tr))
                tally["valid_read"] += used
                tally["correct_scalar"] += scalar
                tally["correct_statement_both"] += both
                tally["correct_final"] += q["score"]["success"]
                if text.strip():
                    literals[canonical(text)] += 1
                by_pair[w["cluster"]].append(
                    {
                        "world": w["world"],
                        "answer": text,
                        "read": used,
                        "correct": q["score"]["success"],
                        "correct_sql_both": both,
                    }
                )
            repeated = under_read = any_program = 0
            for k, versions in by_pair.items():
                (x, y) = sorted(versions, key=lambda v: v["world"])
                same = bool(x["answer"].strip()) and canonical(
                    x["answer"]
                ) == canonical(y["answer"])
                both_read = x["read"] and y["read"]
                repeated += same
                under_read += same and both_read
                any_program += x["correct_sql_both"] or y["correct_sql_both"]
                pair_ledger.append(
                    {
                        "arm": a,
                        "source": k,
                        "both_read": both_read,
                        "same_nonempty_answer": same,
                        "original_answer": x["answer"],
                        "altered_answer": y["answer"],
                        "both_final_correct": x["correct"] and y["correct"],
                        "same_query_correct_on_both_found_in_either_run": x[
                            "correct_sql_both"
                        ]
                        or y["correct_sql_both"],
                    }
                )
            assert tally["cases"] == 48 and len(by_pair) == 24
            summary[a] = {
                **tally,
                "distinct_nonempty_replies": len(literals),
                "top_reply_count": max(literals.values(), default=0),
                "top_reply": literals.most_common(1)[0][0] if literals else "",
                "identical_nonempty_pairs": repeated,
                "identical_after_two_reads": under_read,
                "whole_sources_with_general_query": any_program,
            }
            curves.extend(
                {"arm": a, "per_turn_cap": k, **v} for k, v in cap_totals.items()
            )
    write_csv(P / "conditional-primary-changed-statements.csv", details)
    write_csv(P / "conditional-primary-changed-pairs.csv", pair_ledger)
    write_csv(P / "conditional-primary-output-caps.csv", curves)
    out = {
        "comparison": digest(ROOT / "checksums.json"),
        "summary": summary,
        "caps": curves,
    }
    save(P / "conditional-primary-tool-transfer.json", out)
    return out


if __name__ == "__main__":
    build()
