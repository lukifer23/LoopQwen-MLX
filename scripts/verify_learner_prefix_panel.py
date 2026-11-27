"""Reconstruct the 48 original openings and execute every new finance reference."""

import json
import re
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile

from loopqwen import qwen_tools
from loopqwen.calculator_tool import calculate
from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.research.cross_view_study import tokenizer
from scripts.learner_prefix_panel import PANEL, PLAN
from scripts.shared_query_material import source
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed


def run():
    committed(PANEL)
    committed(PLAN)
    enrolled = unpack(PANEL)["rows"]
    assert digest(PANEL) == read(PLAN)["material"]
    raw_path = Path("data/research_external/tatqa/tatqa_dataset_dev.json")
    raw = {q["uid"]: (q, d) for d in read(raw_path) for q in d["questions"]}
    with ZipFile(source("strategyqa/strategyqa_dataset.zip")) as z:
        questions = {r["qid"]: r for r in json.loads(z.read("strategyqa_train.json"))}
        passages = json.loads(z.read("strategyqa_train_paragraphs.json"))
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    table_uids = set()
    topics = set()
    used_paras = set()
    report = []
    for w in enrolled:
        prompt = w["questions"][0]
        ids = tok.encode(
            qwen_tools.render(tok, [{"role": "user", "content": prompt}], w["defs"]),
            add_special_tokens=False,
        )
        assert ids == w["initial_ids"] and fingerprint(ids) == w["initial_sha256"]
        family, uid = w["id"].split("/")
        if family == "strategyqa":
            q = questions[uid]
            assert q["term"] not in topics and not used_paras.intersection(
                w["paragraphs"]
            )
            assert (
                q["facts"] == w["facts"]
                and q["answer"] == w["label"]
                and q["decomposition"] == w["decomposition"]
            )
            natural = [
                passages[k]["title"] + "\n" + passages[k]["content"]
                for k in w["paragraphs"]
            ]
            assert prompt == "\n\n".join(natural + [q["question"].strip()])
            topics.add(q["term"])
            used_paras.update(w["paragraphs"])
            continue
        q, d = raw[uid]
        table = "\n".join(" | ".join(row) for row in d["table"]["table"])
        body = table + "\n\n" + "\n\n".join(p["text"] for p in d["paragraphs"])
        assert (
            prompt == body + "\n\n" + q["question"]
            and d["table"]["uid"] == w["cluster"]
        )
        assert w["program"] == q["derivation"] and w["cluster"] not in table_uids
        table_uids.add(w["cluster"])
        actual = calculate(w["steps"])
        assert actual == w["observed"] and abs(
            Fraction(actual["value"]) - Fraction(str(q["answer"]))
        ) <= Fraction(1, 200)
        # Ignore conventional expression constants only if arithmetic and wording explain them.
        nums = r"\d+(?:\.\d+)?"
        operands = set(map(Fraction, re.findall(nums, w["steps"][0])))
        printed = set(
            map(
                Fraction,
                re.findall(nums, re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", body)),
            )
        )
        missing = operands - printed
        assert not missing or (
            missing == {2}
            and "average" in q["question"].lower()
            and re.search(r"/\s*2\s*$", w["steps"][0])
        )
        report.append(
            {
                "id": w["id"],
                "report": w["cluster"],
                "source_operands": [str(v) for v in sorted(operands & printed)],
                "averaging_divisors": [str(v) for v in sorted(missing)],
                "returned": actual["value"],
                "reference": str(q["answer"]),
                "scale": q["scale"],
            }
        )
    assert len(report) == len(topics) == len(table_uids) == 24
    persist(
        Path("docs/measurements/learner-prefix-panel.json"),
        {
            "panel": digest(PANEL),
            "source_sha256": digest(raw_path),
            "openings": len(enrolled),
            "reasoning_topics": len(topics),
            "distinct_reasoning_passages": len(used_paras),
            "financial_tables": len(table_uids),
            "programs": report,
        },
    )


if __name__ == "__main__":
    run()
