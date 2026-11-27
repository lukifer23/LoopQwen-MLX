"""Reconstruct every new test opening and every enrolled training citation from original releases."""

from collections import Counter
from fractions import Fraction
from pathlib import Path

import pyarrow.parquet as pq

from loopqwen import qwen_tools
from loopqwen.calculator_tool import calculate
from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.tool_session import compact
from loopqwen.research.cross_view_study import tokenizer
from scripts.content_read_material import CACHE, DIR, HOT, FIN, windows
from scripts.register_content_read import PANEL, TARGETS
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed


def run():
    for p in (PANEL, TARGETS, DIR / "inventory.json"):
        committed(p)
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    held = unpack(PANEL)["rows"]
    train = unpack(DIR / "training.json.gz")
    lookup = {w["id"]: w for w in held}
    for r in train:
        k = r["id"].rsplit("/", 1)[0]
        if k not in lookup:
            lookup[k] = r
    requests = {
        k: (w.get("request") or w["messages"][0]["content"]) for k, w in lookup.items()
    }
    counts = Counter()
    held_titles = set()
    train_titles = set()
    seen = set()
    for key, sha in HOT.items():
        path = CACHE / "hotpotqa" / (key + ".parquet")
        assert digest(path) == sha
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for q in batch.to_pylist():
                k = "hotpotqa/" + q["id"]
                if k not in lookup:
                    continue
                w = lookup[k]
                ctx = q["context"]
                titlemap = dict(zip(ctx["title"], ctx["sentences"], strict=True))
                src = (
                    "\n\n".join(
                        t + "\n" + " ".join(v.strip() for v in ss)
                        for t, ss in titlemap.items()
                    )
                    + "\n\n"
                    + q["question"]
                )
                assert src == requests[k]
                sf = q["supporting_facts"]
                facts = [
                    titlemap[t][i].strip()
                    for t, i in zip(sf["title"], sf["sent_id"], strict=True)
                ]
                gold = q["answer"].capitalize() + ". " + " ".join(facts)
                if "family" in w:
                    assert (
                        key.startswith("validation")
                        and w["reference"] == gold
                        and w["facts"] == facts
                    )
                    held_titles.update(titlemap)
                else:
                    assert key.startswith("train") and w["target"]["content"] == gold
                    assert [src[a:b] for a, b in w["anchors"]] == facts
                    train_titles.update(titlemap)
                seen.add(k)
                counts["hotpot_questions"] += 1
    assert held_titles.isdisjoint(train_titles)
    for split, sha in FIN.items():
        path = CACHE / "tatqa" / ("tatqa_dataset_" + split + ".json")
        assert digest(path) == sha
        for d in read(path):
            table = "\n".join(" | ".join(cells) for cells in d["table"]["table"])
            for q in d["questions"]:
                k = "tatqa/" + q["uid"]
                if k not in lookup:
                    continue
                w = lookup[k]
                src = (
                    table
                    + "\n\n"
                    + "\n\n".join(x["text"] for x in d["paragraphs"])
                    + "\n\n"
                    + q["question"]
                )
                assert src == requests[k]
                if split == "dev":
                    assert "family" in w and w["program"] == q["derivation"]
                    actual = calculate(w["steps"])
                    assert w["observed"] == actual
                else:
                    assert "messages" in w
                    act = next(r for r in train if r["id"] == k + "/action")
                    actual = calculate(
                        act["target"]["tool_calls"][0]["function"]["arguments"]["steps"]
                    )
                    rsp = next(r for r in train if r["id"] == k + "/reply")
                    assert (
                        rsp["target"]["content"]
                        == (str(q["answer"]) + " " + q["scale"]).strip()
                    )
                    assert rsp["messages"][-1]["content"] == compact(actual)
                assert abs(
                    Fraction(actual["value"]) - Fraction(str(q["answer"]))
                ) <= Fraction(1, 200)
                seen.add(k)
                counts["executed_financial_questions"] += 1
    assert seen == lookup.keys()
    # Check every actual training user message including retained and SQL examples against held source documents.
    hold_words = set().union(*(windows(w["request"]) for w in held))
    bound = unpack(TARGETS)
    tokens = 0
    citations = 0
    for w in bound:
        assert not (windows(tok.decode(w["p"])) & hold_words)
        p = w["p"]
        y = w["y"]
        tokens += len(p)
        assert y[-1] == tok.convert_tokens_to_ids("<|im_end|>")
        for mark in w["annotations"]:
            a, b = mark["source"][0]
            assert 0 <= a < b <= len(p) and 0 <= mark["query"] < len(y)
            citations += 1
    for w in held:
        ids = tok.encode(
            qwen_tools.render(
                tok, [{"role": "user", "content": w["request"]}], w["defs"]
            ),
            add_special_tokens=False,
        )
        assert ids == w["initial_ids"] and fingerprint(ids) == w["initial_sha256"]
    # SQL schema boilerplate and assistant tool definitions are outside source-text overlap.
    assert not any(windows(r["messages"][0]["content"]) & hold_words for r in train)
    out = dict(
        held=digest(PANEL),
        training=digest(TARGETS),
        counts=dict(counts),
        openings=len(held),
        training_turns=len(bound),
        training_opening_tokens=tokens,
        source_citations=citations,
        held_titles=len(held_titles),
        train_titles=len(train_titles),
        shared_titles=0,
    )
    persist(Path("docs/measurements/content-read-sources.json"), out)
    print(out)


if __name__ == "__main__":
    run()
