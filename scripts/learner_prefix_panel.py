"""Reserve source-disjoint human reasoning and executable financial questions for one direct pair."""

import json
from collections import Counter
from fractions import Fraction
import hashlib
import re
from urllib.request import urlopen
from pathlib import Path
from zipfile import ZipFile

from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.qwen_tools import render
from loopqwen.research.cross_view_study import tokenizer
from scripts.learner_prefix_material import DATA, DIR, SEED
from scripts.shared_query_material import (
    ROOT,
    TOOLS,
    fits,
    CALCULATOR,
    source,
    strategy_rows,
)
from scripts.table_policy_material import persist

PANEL = DIR / "fresh-panel.json.gz"
PLAN = Path("research/diagnostics/learner-prefix-direct.json")


def build():
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    prior = unpack(ROOT / "held.json.gz")
    old = unpack(DATA)
    old_fin = read("research/data/report-calculation-panel.json")
    finance_used = {r["cluster"] for r in old_fin["rows"]} | set(
        old_fin["held_out_issuers"]
    )
    finance_used.update(r["cluster"] for r in old if r["group"].startswith("report_"))
    finance_used.update(r["cluster"] for r in prior if r["family"] == "report")
    training_ids = {r["id"].split("/")[-1] for r in old if r["group"] == "strategy"}
    training_ids.update(
        r["id"].split("/")[-1] for r in prior if r["family"] == "strategy"
    )
    topics = set()
    paragraphs = set()
    held = []

    def add(item, family, tools):
        message = {"role": "user", "content": item["request"]}
        p = tok.encode(render(tok, [message], tools), add_special_tokens=False)
        held.append(
            {
                **item,
                "family": family,
                "defs": tools,
                "questions": [item["request"]],
                "initial_ids": p,
                "initial_sha256": fingerprint(p),
            }
        )

    with ZipFile(source("strategyqa/strategyqa_dataset.zip")) as z:
        raw = json.loads(z.read("strategyqa_train.json"))
        passages = json.loads(z.read("strategyqa_train_paragraphs.json"))
    for r in raw:
        if r["qid"] in training_ids:
            _, w = strategy_rows(r, passages)
            topics.add(w["cluster"])
            paragraphs.update(w["paragraphs"])
    original_topics = len(topics)
    original_paras = len(paragraphs)
    labels = Counter()
    for r in sorted(raw, key=lambda x: fingerprint([SEED, "fresh topic", x["qid"]])):
        if r["term"] in topics or labels[r["answer"]] >= 12:
            continue
        try:
            rows, w = strategy_rows(r, passages)
        except ValueError:
            continue
        if paragraphs.intersection(w["paragraphs"]) or not fits(tok, rows):
            continue
        add(w, "strategy", [])
        labels[r["answer"]] += 1
        topics.add(w["cluster"])
        paragraphs.update(w["paragraphs"])
        if len(held) == 24:
            break
    assert labels == {False: 12, True: 12}
    # TAT-QA development provides independent published reports and a worked calculation.
    # Exclude every previously enrolled TAT-QA report, plus shared 16-word passages
    # with earlier financial training/test sources, before ordering the new contexts.
    old_tat = read("research/data/report-action-training.json")
    used = {r["id"] for r in old_tat["pairs"]}
    used.update(
        r["cluster"]
        for r in read("research/data/report-action-panel.json")["rows"]
        if r["family"] == "report"
    )

    def shingles(text):
        words = re.findall(r"[a-z]+|[0-9]+", text.casefold())
        return {" ".join(words[i : i + 16]) for i in range(len(words) - 15)}

    seen = set()
    for r in old:
        if r["group"] == "report_action":
            seen.update(shingles(r["messages"][0]["content"]))
    for r in prior:
        if r["family"] == "report":
            seen.update(shingles(r["request"]))
    for r in old_tat["examples"]:
        for m in r["messages"]:
            if m["role"] == "user":
                seen.update(shingles(m["content"]))
    for r in read("research/data/report-action-panel.json")["rows"]:
        if r["family"] == "report":
            seen.update(shingles(r["questions"][0]))
    source_path = Path("data/research_external/tatqa/tatqa_dataset_dev.json")
    if not source_path.exists():
        source_path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(
            "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/870accc41953dcde885aabeb963d94aabdc0fbc3/dataset_raw/tatqa_dataset_dev.json",
            timeout=60,
        ) as response:
            source_path.write_bytes(response.read())
    assert (
        digest(source_path)
        == "8da095a819af6db3c14877c6df2d4d29960e41d1a63dd1fa853507bd2a616af5"
    )
    for doc in sorted(
        read(source_path),
        key=lambda x: fingerprint([SEED, "financial context", x["table"]["uid"]]),
    ):
        table = "\n".join(" | ".join(row) for row in doc["table"]["table"])
        body = table + "\n\n" + "\n\n".join(p["text"] for p in doc["paragraphs"])
        if hashlib.sha256(body.encode()).hexdigest() in used or seen.intersection(
            shingles(body)
        ):
            continue
        for q in sorted(
            doc["questions"], key=lambda x: fingerprint([SEED, "arith", x["uid"]])
        ):
            if q["answer_type"] != "arithmetic":
                continue
            expr = q["derivation"].replace("−", "-").replace("×", "*").replace("÷", "/")
            expr = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", expr)
            if not re.search(r"\d\s*[+*/-]\s*[(\d]", expr):
                continue
            try:
                gold = Fraction(str(q["answer"]))
                steps = [expr]
                val = CALCULATOR.callback(steps=steps)
                if q["scale"] == "percent" and abs(
                    Fraction(val["value"]) - gold
                ) > Fraction(1, 200):
                    steps.append("r1*100")
                    val = CALCULATOR.callback(steps=steps)
                if abs(Fraction(val["value"]) - gold) > Fraction(1, 200):
                    continue
            except (ValueError, TypeError, ZeroDivisionError, SyntaxError):
                continue
            prompt = body + "\n\n" + q["question"]
            if (
                len(
                    tok.encode(
                        render(tok, [{"role": "user", "content": prompt}], TOOLS),
                        add_special_tokens=False,
                    )
                )
                > 1536
            ):
                continue
            w = dict(
                id="tatqa/" + q["uid"],
                cluster=doc["table"]["uid"],
                source="TAT-QA development",
                request=prompt,
                question=q["question"],
                reference=str(q["answer"]) + " " + q["scale"],
                gold=str(gold),
                tolerance="1/200",
                program=q["derivation"],
                steps=steps,
                observed=val,
                evidence=[table]
                + [
                    p["text"]
                    for p in doc["paragraphs"]
                    if p["order"] in q["rel_paragraphs"]
                ],
            )
            add(w, "report", TOOLS)
            seen.update(shingles(body))
            break
        if len(held) == 48:
            break
    assert len(held) == 48 and len({w["id"] for w in held}) == 48
    held.sort(key=lambda w: fingerprint([SEED, "fresh comparison", w["id"]]))
    material = dict(
        rows=held,
        prior_topics=original_topics,
        prior_passages=original_paras,
        prior_issuers=sorted(finance_used),
    )
    if PANEL.exists():
        assert unpack(PANEL) == material
    else:
        packed(PANEL, material)
    persist(
        PLAN,
        dict(
            material=digest(PANEL),
            predecessor=digest(
                Path("research/results/staged-query-transfer-01/checksums.json")
            ),
            question="Can corrective training from actual learner prefixes improve source-grounded reasoning and executable calculations on previously unused topics, passages and financial tables?",
            design="One direct pair on 24 new human topics and 24 previously unused TAT-QA financial-report contexts. Both get identical original opening IDs, official thinking mode, greedy decoding, 384 tokens per turn, up to three turns and identical calculator callbacks. Report strict correctness, every gain and loss, actual wall time, first-output wait, peak MLX memory and both token rates. Blindly judge every source response before opening per-arm outcomes. The candidate is the prespecified two-read fit. Control is untouched certified BF16 Qwen Base at ordinary depth. No recovery suffix, forced token, answer verifier or deadline heuristic is available to either at inference.",
            gate="Within each family require at least six additional correct and fully-usable source answers on their respective endpoints, zero lost successes, no additional unsupported claims and two-sided exact paired p<=0.05. The financial endpoint requires an executed substantive calculation and a correct delivered quantity. Require superior aggregate elapsed, first-token wait, opening throughput, decode throughput and peak, and no fewer completed answers within either family. All 48 pairs are retained. No later confirmation or release from a failed comparison. At a pass, test external tool error and misleading histories, broad retained tasks, full-length edited sources, matched trained depth and an independent seed.",
            prerequisites="Commit a successful actual-prefix collection and a verified fit with fixed two physical reads, positive source and executed-action target improvements and protected language retention. Train only on existing training data. All earlier source-repair topics and reasoning paragraphs and all previously enrolled TAT-QA reports are excluded. No new finance context shares a normalized 16-word span with the preceding FinQA training or test or the earlier TAT-QA enrollments. TAT-QA does not supply a reliable issuer key, so issuer independence is unknown. Public donor pretraining exposure is unknown.",
            limits={
                "seconds_per_arm": 1200,
                "total_seconds": 2400,
                "peak_gib": 3.0,
                "available_gib": 1.5,
            },
            ordered=[w["id"] for w in held],
        ),
    )
    print(
        len(held),
        "questions;",
        original_topics,
        "earlier topics,",
        original_paras,
        "earlier paragraphs and",
        len(finance_used),
        "tracked FinQA issuers; new financial cases from TAT-QA",
    )


if __name__ == "__main__":
    build()
