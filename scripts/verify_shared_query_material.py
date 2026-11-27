"""Reconstruct every enrolled human source, argument and tool result without model inference."""

import json
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from zipfile import ZipFile

from loopqwen.calculator_tool import calculate
from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from scripts.shared_query_material import ROOT, source


def norm(s):
    return " ".join(s.casefold().split())


def verify():
    m = read(ROOT / "source-inventory.json")
    train, held = [unpack(ROOT / (n + ".json.gz")) for n in ("training", "held")]
    assert all(
        m[n + "_sha256"] == digest(ROOT / (n + ".json.gz"))
        for n in ("training", "held")
    )
    with ZipFile(source("strategyqa/strategyqa_dataset.zip")) as z:
        strategy = {r["qid"]: r for r in json.loads(z.read("strategyqa_train.json"))}
        paragraphs = json.loads(z.read("strategyqa_train_paragraphs.json"))
    finance = {
        split: {r["id"]: r for r in read(source("finqa/" + split + ".json"))}
        for split in ("train", "test")
    }
    topics, companies, pages = set(), set(), set()
    questions = {}
    training_requests = set()
    stored_actions = {}
    held_questions = {}
    counted = Counter()

    def facts(k, context):
        r = strategy[k.split("/", 1)[1]]
        q = r["question"].strip()
        assert context.endswith("\n\n" + q)
        texts = context[: -(len(q) + 2)].split("\n\n")
        # Paragraph prose may itself contain blank lines. Verify source blocks by their full original text.
        atoms = []

        def visit(v):
            if isinstance(v, list):
                for part in v:
                    visit(part)
            elif v in paragraphs:
                p = paragraphs[v]
                block = p["title"] + "\n" + p["content"]
                if block in context:
                    atoms.append((context.index(block), v, block))

        visit(r["evidence"])
        ordered = sorted(set(atoms))
        assembled = "\n\n".join(dict.fromkeys(b for _, _, b in ordered))
        assert assembled + "\n\n" + q == context and texts
        return r, {ident for _, ident, _ in ordered}

    for r in train:
        request = r["messages"][0]["content"]
        training_requests.add(fingerprint(request))
        if r["group"] == "strategy":
            raw, ids = facts(r["id"], request)
            assert r["cluster"] == raw["term"]
            assert r["target"]["content"] == (
                "Yes. " if raw["answer"] else "No. "
            ) + " ".join(raw["facts"])
            assert r["target"]["reasoning_content"].splitlines() == [
                f"{n}. {q}" for n, q in enumerate(raw["decomposition"], 1)
            ]
            topics.add(raw["term"])
            pages.update(ids)
            questions[norm(raw["question"])] = r["id"]
            counted["reconstructed_strategy"] += 1
        if r["group"].startswith("report_"):
            k = r["id"].removeprefix("finqa/").rsplit("/", 1)[0]
            raw = finance["train"][k]
            company = raw["filename"].split("/")[0]
            companies.add(company)
            q = raw["qa"]
            assert company == r["cluster"] and request.endswith(q["question"])
            questions[norm(q["question"])] = r["id"]
            message = r["target"] if r["group"] == "report_action" else r["messages"][1]
            (a,) = message["tool_calls"]
            assert a["function"]["name"] == "calculate"
            result = calculate(**a["function"]["arguments"])
            recorded = (fingerprint(a), fingerprint(result))
            assert recorded == stored_actions.setdefault(k, recorded)
            if r["group"] == "report_reply":
                assert json.loads(r["messages"][2]["content"]) == result
                assert r["target"]["content"] == str(q["answer"]).strip()
            counted[r["group"]] += 1
    for item in held:
        q = item["question"]
        held_questions[item["id"]] = norm(q)
        assert fingerprint(item["request"]) not in training_requests
        if item["family"] == "strategy":
            raw, ids = facts(item["id"], item["request"])
            assert raw["term"] not in topics and ids.isdisjoint(pages)
            assert raw["facts"] == item["facts"] and raw["answer"] == item["label"]
            assert set(item["paragraphs"]) == ids
        else:
            raw = finance["test"][item["id"].removeprefix("finqa/")]
            assert (
                raw["filename"] == item["source"] and item["cluster"] not in companies
            )
            assert (
                raw["qa"]["question"] == q
                and str(raw["qa"]["answer"]).strip() == item["reference"]
            )
            result = calculate(item["steps"])
            assert result == item["observed"]
            frac = Fraction(result["value"])
            v = Decimal(frac.numerator) / Decimal(frac.denominator)
            assert abs(v - Decimal(item["gold"])) <= Decimal(item["tolerance"])
        counted["held_" + item["family"]] += 1
    result = dict(
        verified=dict(counted),
        unique_calculator_trajectories=len(stored_actions),
        held_full_prompts_found_in_train=0,
        shared_strategy_topics=len(topics & set(m["held_topics"])),
        shared_strategy_paragraph_ids=len(pages & set(m["held_paragraph_ids"])),
        shared_finance_issuers=len(companies & set(m["held_issuers"])),
        repeated_question_wordings={
            k: questions[q] for k, q in held_questions.items() if q in questions
        },
        inventory_sha256=digest(ROOT / "source-inventory.json"),
    )
    save(ROOT / "independent-replay.json", result)
    print(result)


if __name__ == "__main__":
    verify()
