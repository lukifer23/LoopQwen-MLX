"""Locate the registered output limit within each model's actual generation stream."""

from collections import Counter
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read
from loopqwen.research.cross_view_study import tokenizer
from scripts.query_content_depth import write_csv
from scripts.table_policy_material import persist

ROOT = Path("research/results/staged-query-transfer-01")
OUT = Path("docs/measurements/staged-query-completion.json")


def run():
    cfg = read(ROOT / "registration.json")
    tok = tokenizer(cfg)
    close = tok.convert_tokens_to_ids("</think>")
    rows = []
    groups = {}
    for arm in ("candidate", "control"):
        for item in unpack(ROOT / (arm + ".json.gz")):
            tr = item["requests"][0]["trajectory"]
            ids = tr["turns"][-1]["token_ids"]
            count = ids.count(close)
            cut = tr["status"] == "token_limit"
            before = ids.index(close) if count else len(ids)
            grams = list(zip(ids, ids[1:], ids[2:], ids[3:]))
            repeated = sum(n - 1 for n in Counter(grams).values())
            r = {
                "arm": arm,
                "id": item["id"],
                "family": item["family"],
                "status": tr["status"],
                "last_turn_tokens": len(ids),
                "first_close": before if count else None,
                "close_tokens": count,
                "repeated_fourgrams": repeated,
                "total_fourgrams": len(grams),
                "at_limit_inside_thought": cut and not count,
                "at_limit_after_close": cut and bool(count),
            }
            if cut:
                assert len(ids) == cfg["max_tokens"]
            else:
                assert tr["status"] != "answered" or count > 0
            rows.append(r)
            c = groups.setdefault(item["family"], {}).setdefault(arm, Counter())
            c["requests"] += 1
            c[tr["status"]] += 1
            c["at_limit_inside_thought"] += r["at_limit_inside_thought"]
            c["at_limit_after_close"] += r["at_limit_after_close"]
            if cut:
                c["repeated_fourgrams"] += repeated
                c["total_fourgrams"] += len(grams)
    assert len(rows) == 360
    write_csv(Path("paper/staged-query-completion.csv"), rows)
    persist(
        OUT,
        {
            "archive": digest(ROOT / "checksums.json"),
            "groups": groups,
            "scope": "Post hoc mechanism diagnosis after the blind reading. Count the first actual closing thought token in the last turn of each request and whether the registered 384-token limit was reached before or after it. Overlapping fourgrams beyond their first occurrence quantify local repetition. No partial or hidden text is credited as a delivered answer.",
        },
    )


if __name__ == "__main__":
    run()
