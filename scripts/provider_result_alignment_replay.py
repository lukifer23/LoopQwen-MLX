"""Rebuild both SQL calls and every serialized successful or corrective history from source tables."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.relational_tools import tools
from loopqwen.research.table_worlds import FILENAME, canonical, mount, prompt
from loopqwen.tool_session import compact
from scripts.provider_recovery_csv import ENROLLED, MULTI
from scripts.provider_recovery_material import ROOT
from scripts.provider_recovery_replay import session
from scripts.table_policy_material import persist


def replay():
    originals = {w["id"]: w for w in unpack(MULTI)}
    q = read(ROOT / "semantic-qualification.json")
    corpus = ROOT / "semantic-training.json.gz"
    assert (q["source"], q["eligible"], q["training_sha256"]) == (
        digest(MULTI),
        digest(ENROLLED),
        digest(corpus),
    )
    rows = unpack(corpus)
    by = defaultdict(dict)
    for row in rows:
        pair = (row["source"], row["world"])
        assert row["group"] not in by[pair]
        by[pair][row["group"]] = row
    counts = Counter()
    consumed = set()
    with TemporaryDirectory() as t:
        for pair in q["origins"]:
            s1, s2 = pair["sources"]
            for i, (k, other) in enumerate(((s1, s2), (s2, s1))):
                w = originals[k]
                own, wrong = (originals[z]["statement"] for z in (k, other))
                assert (
                    own == pair["programs"][i]
                    and w["context"] == originals[other]["context"] == pair["context"]
                )
                for v, fact in zip(w["versions"], pair["checked"][i], strict=True):
                    assert (
                        v["world"] == fact["world"] and v["answer"] == fact["correct"]
                    )
                    db = mount(Path(t) / k / str(v["world"]), v["rows"])
                    ps = {p.name: p for p in tools([db])}
                    rd = ps["database_query"].callback
                    results = [
                        rd(database_name=FILENAME, statement=s) for s in (own, wrong)
                    ]
                    for got in results:
                        assert (
                            got["referenced_tables"] == ["records"]
                            and not got["truncated"]
                        )
                        assert (
                            len(got["rows"]) == len(got["rows"][0]) == 1
                            and got["rows"][0][0] is not None
                        )
                    y, n = (z["rows"][0][0] for z in results)
                    assert canonical(y) == canonical(v["answer"]) and canonical(
                        n
                    ) != canonical(y)
                    assert n == fact["different"] and [
                        fingerprint(z) for z in results
                    ] == [fact["own_hash"], fact["other_hash"]]
                    if not v["world"]:
                        assert canonical(n) == canonical(
                            originals[other]["versions"][0]["answer"]
                        )
                    counts["real_distinct_scalar_pairs"] += 1
                    key = (k, v["world"])
                    if key not in by:
                        continue
                    group = by[key]
                    assert set(group) == {
                        "answer_valid_read",
                        "redirect_valid_read",
                        "answer_after_redirect",
                    }
                    redirect = group["redirect_valid_read"]
                    delivered = group["answer_after_redirect"]
                    accepted = group["answer_valid_read"]
                    assert redirect["messages"] == delivered["messages"][:-2]
                    assert accepted["messages"][0]["content"] == prompt(w)
                    assert accepted["messages"][:3] == redirect["messages"][:3]
                    assert accepted["messages"][-2:] == delivered["messages"][-2:]
                    assert delivered["messages"][-2] == redirect["target"]
                    assert accepted["target"] == delivered["target"] and canonical(
                        accepted["target"]["content"]
                    ) == canonical(v["answer"])
                    for row in group.values():
                        assert row["other"] == other and row["source"] == k
                        msgs = list(row["messages"])
                        if row["group"] == "redirect_valid_read":
                            assert json.loads(msgs[-1]["content"]) == results[1]
                            msgs.extend(
                                (
                                    row["target"],
                                    {"role": "tool", "content": compact(results[0])},
                                )
                            )
                        else:
                            assert json.loads(msgs[-1]["content"]) == results[0]
                        counts.update(session(msgs, ps))
                        consumed.add(row["id"])
            for own, other in ((s1, s2), (s2, s1)):
                pos = by[own, 0]["answer_valid_read"]
                neg = by[other, 0]["redirect_valid_read"]
                assert (
                    pos["messages"][0] != neg["messages"][0]
                    and pos["messages"][1:] == neg["messages"][1:]
                )
                assert (
                    pos["tools"] == neg["tools"]
                    and "tool_calls" not in pos["target"]
                    and "tool_calls" in neg["target"]
                )
                counts[
                    "identical_call_and_observation_with_opposite_required_behavior"
                ] += 1
    assert len(consumed) == len(rows) and not counts["error"]
    result = {
        "training": digest(corpus),
        "origin_inventory": digest(ROOT / "semantic-qualification.json"),
        "verified_turns": len(consumed),
        **dict(counts),
    }
    persist(ROOT / "semantic-independent-replay.json", result)
    print(result)


if __name__ == "__main__":
    replay()
