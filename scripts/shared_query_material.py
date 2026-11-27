"""Enroll complete source-backed reasoning and executable report work for shared-query learning."""

import json
import re
from collections import Counter
from decimal import InvalidOperation
from fractions import Fraction
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from loopqwen import qwen_tools
from loopqwen.calculator_tool import tool
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from scripts.program_language_material import answer, call, observed, user
from scripts.report_calculation_material import FIN, OPERATORS, finance, scalar

ROOT = Path("research/sources/shared-query")
DATA = Path("data/research_external")
STRATEGY = "https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip"
FINANCE = f"https://raw.githubusercontent.com/czyssrs/FinQA/{FIN}/dataset/"
RAW = {
    "strategyqa/strategyqa_dataset.zip": (
        STRATEGY,
        "4911d85eb6721a93bed7645419df77e721808b32b9785dee14ad80e6249e0a90",
    ),
    "finqa/train.json": (
        FINANCE + "train.json",
        "49f237eb9779b569473b26b08048867d04635a7cc39ad6a7a5664c55bb428db6",
    ),
    "finqa/test.json": (
        FINANCE + "test.json",
        "831dbfb2e785dbc227f895ce3f24046433467aec67b09db2bd6ac7692a8a30dc",
    ),
}
SEED = 864371
MAX_SEQUENCE = 1536
MAX_TARGET = 384
CALCULATOR = tool()
TOOLS = qwen_tools.declarations([CALCULATOR])
PARENTS = ("program-language", "contrastive-worlds")


def source(name):
    url, sha = RAW[name]
    p = DATA / name
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url, timeout=100) as res:
            raw = res.read()
        import hashlib

        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError(f"Changed upstream source {name}")
        p.write_bytes(raw)
    if digest(p) != sha:
        raise ValueError(f"Source digest mismatch for {p}")
    return p


def priority(key):
    return fingerprint([SEED, key])


def row(key, group, cluster, messages, tools, target):
    return dict(
        id=key,
        group=group,
        cluster=cluster,
        messages=messages,
        tools=tools,
        target=target,
    )


def numeric_argument(arg):
    if arg.startswith("#"):
        return "r" + str(int(arg[1:]) + 1)
    s = arg.removeprefix("const_")
    if arg.startswith("const_m"):
        s = "-" + arg[7:]
    v = Fraction(scalar(s))
    return str(v / 100 if s.endswith("%") else v)


def report_rows(raw):
    w = finance(raw)
    qa = raw["qa"]
    expr = []
    for op, a, b in re.findall(r"(\w+)\(([^()]+), ([^()]+)\)", qa["program"]):
        expr.append(f"({numeric_argument(a)}){OPERATORS[op]}({numeric_argument(b)})")
    if len(w["steps"]) > len(expr):
        assert len(w["steps"]) == len(expr) + 1
        expr.append(f"r{len(expr)}*100")
    result = CALCULATOR.callback(steps=expr)
    prior = [CALCULATOR.callback(steps=[v])["value"] for v in w["steps"]]
    assert [Fraction(v) for v in result["results"]] == [Fraction(v) for v in prior]
    # The actual request contains the report and the expert's question, never gold evidence or steps.
    prompt = w["context"] + "\n\n" + w["question"]
    evidence = " ".join(dict.fromkeys(w["evidence"])).strip()
    if not evidence:
        raise ValueError("Missing expert supporting text")
    key = "finqa/" + w["id"]
    messages = [user(prompt)]
    action = call(CALCULATOR, steps=expr)
    action["reasoning_content"] = evidence
    target = str(qa["answer"]).strip()
    rows = [
        row(key + "/action", "report_action", w["cluster"], messages, TOOLS, action),
        row(
            key + "/reply",
            "report_reply",
            w["cluster"],
            [*messages, action, observed(result)],
            TOOLS,
            answer(target),
        ),
    ]
    return rows, dict(
        id=key,
        cluster=w["cluster"],
        source=w["report"],
        request=prompt,
        question=w["question"],
        reference=target,
        gold=w["gold"],
        tolerance=w["tolerance"],
        program=qa["program"],
        steps=expr,
        observed=result,
        evidence=w["evidence"],
    )


def paragraph_ids(value, inventory):
    if isinstance(value, str):
        return [value] if value in inventory else []
    return [k for part in value for k in paragraph_ids(part, inventory)]


def strategy_rows(raw, inventory):
    complete = [
        v
        for v in raw["evidence"]
        if len(v) == len(raw["decomposition"])
        and all(paragraph_ids(step, inventory) or "operation" in step for step in v)
    ]
    if not complete or len(raw["facts"]) < 2 or len(raw["decomposition"]) < 2:
        raise ValueError("Incomplete supporting annotation")
    choices = [list(dict.fromkeys(paragraph_ids(v, inventory))) for v in complete]
    choices = [keys for keys in choices if keys]
    if not choices:
        raise ValueError("No cited passages")
    # Select a complete annotator trace by original IDs without adding gold facts to the input.
    keys = min(
        choices,
        key=lambda k: (sum(len(inventory[i]["content"]) for i in k), priority(k)),
    )
    sections = [inventory[k]["title"] + "\n" + inventory[k]["content"] for k in keys]
    question = raw["question"].strip()
    visible = ("Yes. " if raw["answer"] else "No. ") + " ".join(raw["facts"])
    plan = "\n".join(f"{i}. {text}" for i, text in enumerate(raw["decomposition"], 1))
    ident = "strategyqa/" + raw["qid"]
    request = "\n\n".join([*sections, question])
    return [
        row(ident, "strategy", raw["term"], [user(request)], [], answer(visible, plan))
    ], dict(
        id=ident,
        cluster=raw["term"],
        request=request,
        question=question,
        reference=visible,
        label=raw["answer"],
        paragraphs=keys,
        facts=raw["facts"],
        decomposition=raw["decomposition"],
    )


def fits(tok, rows):
    for r in rows:
        p, y = token_pair(tok, r)
        if len(p) + len(y) > MAX_SEQUENCE or len(y) > MAX_TARGET:
            return False
    return True


def build():
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    with ZipFile(source("strategyqa/strategyqa_dataset.zip")) as z:
        science = json.loads(z.read("strategyqa_train.json"))
        passages = json.loads(z.read("strategyqa_train_paragraphs.json"))
    finance_old = read(Path("research/data/report-calculation-panel.json"))
    excluded_issuers = set(finance_old["held_out_issuers"])
    held_fin, held_science, held_paras = set(), set(), set()
    held, training = [], []
    selected = Counter()
    excluded = Counter()
    for split in ("test", "train"):
        limit = 32 if split == "test" else 512
        for raw in sorted(
            read(source(f"finqa/{split}.json")), key=lambda r: priority(r["id"])
        ):
            ticker = raw["filename"].split("/")[0]
            if ticker in held_fin or ticker in excluded_issuers:
                continue
            try:
                rows, item = report_rows(raw)
            except (
                ValueError,
                KeyError,
                ZeroDivisionError,
                InvalidOperation,
                IndexError,
                TypeError,
            ):
                excluded["unverified_finance_" + split] += 1
                continue
            if not fits(tok, rows):
                excluded["finance_length_" + split] += 1
                continue
            if split == "test":
                held.append(dict(family="report", **item))
                held_fin.add(ticker)
            else:
                training.extend(rows)
            selected["report_" + split] += 1
            if selected["report_" + split] == limit:
                break
        assert selected["report_" + split] == limit
    ordered = sorted(science, key=lambda r: priority(r["qid"]))
    for split, limit in (("held", 24), ("train", 384)):
        for raw in ordered:
            label = str(raw["answer"])
            key = "strategy_" + split + "_" + label
            if selected[key] >= limit or raw["term"] in held_science:
                continue
            try:
                rows, item = strategy_rows(raw, passages)
            except ValueError:
                excluded["incomplete_strategy_" + split] += 1
                continue
            if set(item["paragraphs"]) & held_paras:
                continue
            if not fits(tok, rows):
                excluded["strategy_length_" + split] += 1
                continue
            if split == "held":
                held_science.add(raw["term"])
                held_paras.update(item["paragraphs"])
                held.append(dict(family="strategy", **item))
            else:
                training.extend(rows)
            selected[key] += 1
        assert (
            selected["strategy_" + split + "_True"]
            == selected["strategy_" + split + "_False"]
            == limit
        )
    rehearsal = Counter()
    for parent in PARENTS:
        for r in sorted(
            unpack(Path("research/sources") / parent / "training.json.gz"),
            key=lambda r: priority(r["id"]),
        ):
            group = r["group"]
            if rehearsal[group] >= 96 or not fits(tok, [r]):
                continue
            training.append(
                row(
                    "rehearsal/" + r["id"],
                    group,
                    r.get("pair", r["id"]),
                    r["messages"],
                    r["tools"],
                    r["target"],
                )
            )
            rehearsal[group] += 1
    assert len({r["id"] for r in training}) == len(training)
    assert not ({r["id"] for r in training} & {r["id"] for r in held})
    assert all(
        r["cluster"] not in held_fin | excluded_issuers
        for r in training
        if r["group"].startswith("report_")
    )
    assert all(
        r["cluster"] not in held_science for r in training if r["group"] == "strategy"
    )
    training.sort(key=lambda r: priority(r["id"]))
    held.sort(key=lambda r: priority(r["id"]))
    groups = {}
    for r in training:
        p, y = token_pair(tok, r)
        c = groups.setdefault(r["group"], Counter())
        c.update(
            turns=1,
            opening=len(p),
            target=len(y),
            with_reasoning=bool(r["target"].get("reasoning_content", "").strip()),
        )
    packed(ROOT / "training.json.gz", training)
    packed(ROOT / "held.json.gz", held)
    save(
        ROOT / "source-inventory.json",
        dict(
            publisher_sources=RAW,
            prior_reserved_issuers=sorted(excluded_issuers),
            seed=SEED,
            input_plus_target_limit=MAX_SEQUENCE,
            target_limit=MAX_TARGET,
            selected=selected,
            groups=groups,
            exclusions=excluded,
            training_sha256=digest(ROOT / "training.json.gz"),
            held_sha256=digest(ROOT / "held.json.gz"),
            held_topics=sorted(held_science),
            held_issuers=sorted(held_fin),
            held_paragraph_ids=sorted(held_paras),
            rehearsal_sources={
                p: digest(Path("research/sources") / p / "training.json.gz")
                for p in PARENTS
            },
        ),
    )
    print(
        len(training),
        "complete native-format targets and",
        len(held),
        "held questions",
        flush=True,
    )


if __name__ == "__main__":
    build()
