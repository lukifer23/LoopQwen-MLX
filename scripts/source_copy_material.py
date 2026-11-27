"""Freeze source-checked multi-turn supervision and genuinely independent conversation openings."""

from collections import Counter
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile
import json

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import read, digest
from loopqwen.qwen_tools import render, declarations
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from scripts.conversational_finance_sources import FILE, SHA, convert
from scripts.conversational_reasoning_sources import CACHE, OUT as COQA
from scripts.conversational_teacher import OUT as TEACHER, DATA, PROVIDER, inputs
from scripts.content_read_material import persist_rows, windows
from scripts.program_language_material import answer, call, observed, user
from scripts.register_content_read import TARGETS
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

ROOT = Path("research/sources/source-copy")
TOKENS = ROOT / "training.json.gz"
PANEL = ROOT / "conversation-panel.json.gz"
STYLE = "Read the passage and answer the reader’s questions."
FIN = "Use the report to answer the financial questions. Use the calculator for arithmetic and state the result with its units."
SEED = 985607
# Each positive numerical trace was checked against the original published report.
# Each retained delivery below uses the executed program and the appropriate display scale.
REVIEW = {
    "UPS": ("The balance increased by 50%.", "Buildings row, 285 against 190."),
    "TFX": (
        None,
        "The source describes a reinvested index investment. The rationale calls it the price.",
    ),
    "HIG": (
        "The annual average was 6,898 in the table’s reporting units.",
        "2014 and 2013 statutory surplus, 7,157 and 6,639.",
    ),
    "INTC": (
        "Capital expenditures increased by about 22.92% relative to 2016.",
        "The report states 11.8 and 9.6 billion.",
    ),
    "WRK": (
        "Segment income increased by about 41.18%.",
        "Total segment income, 1,685.0 versus 1,193.5 million.",
    ),
    "VNO": (
        "The increase was about 6.25% of the original balance.",
        "Redeemable interests, original 1,177,978 and increase 73,650.",
    ),
    "SWKS": (
        "Unrecognized tax benefits increased by about 61.49%.",
        "Balances 32,136 and 19,900 in thousands.",
    ),
    "LLY": (
        None,
        "Both intangibles cells agree. The supplied passage does not explicitly establish the million-dollar scale used by the rationale.",
    ),
    "MS": (
        None,
        "The report specifies future minimum rental commitments. The rationale describes a lease liability instead.",
    ),
    "IQV": (
        None,
        "S&P 500 was 114 at year-end 2013 and 100 in May 2013. The rationale treats the initial May value as the unqualified annual starting value.",
    ),
    "HFC": (
        "The outstanding grants increased by about 39.30%.",
        "Balance rose from 556,186 to 774,788 performance share units.",
    ),
    "ANSS": (
        "Unrecognized tax benefits increased by about 56.61%.",
        "Opening and closing balances 7,928 and 12,416 in thousands.",
    ),
    "EMR": (
        "The available shares were about 11.21 times the number awarded.",
        "159,965 shares available and 12,228 plus 2,038 awards.",
    ),
    "AAP": (
        None,
        "The program mixes 2012 LIFO with 2011 FIFO. The actual 2011 LIFO row is 2,043,158.",
    ),
    "AAPL": (
        "The value of the investment increased by 331%.",
        "Reinvested series begins at 100 and ends at 431.",
    ),
}


# Source-only review before registration or any model generation.
SOURCE_EXCLUSIONS = {
    "Single_AMAT/2013/page_18.pdf-1": "Question asks for fiscal 2014 from a 2013 report. Its annotated amounts are from 2013 and 2012.",
    "Double_AMT/2004/page_81.pdf": "The two projected amortization amounts are for different years than the question asks.",
    "Single_GS/2012/page_165.pdf-3": "The written divisor and program divisor disagree.",
    "Single_VRTX/2003/page_71.pdf-3": "The second question asks for a change over time from two categories at the same date.",
}


def reports(tok, excluded):
    from loopqwen.calculator_tool import calculate

    assert digest(FILE) == SHA
    trained = unpack(DATA)
    old_issuers = set(
        read("research/data/report-calculation-panel.json")["held_out_issuers"]
    )
    old_pages = {w["file"] for w in trained}
    old_pages.update(
        w["report"]
        for w in read("research/data/report-calculation-panel.json")["rows"]
        if w.get("report")
    )
    train_pages = {
        w["filename"] for w in read("data/research_external/finqa/train.json")
    }
    old_pages.update(train_pages)
    for path in (
        "research/sources/shared-query/held.json.gz",
        "research/sources/repair-query/transfer-panel.json.gz",
    ):
        rows = unpack(Path(path))
        for row in rows if isinstance(rows, list) else rows["rows"]:
            if row.get("family") == "report":
                old_pages.add(row["source"])
                old_issuers.add(row["cluster"])
    # Include the actual pages of the earlier FinQA source enrollments.
    for file in Path("research/sources").glob("**/*held*.json.gz"):
        if file.stat().st_size > 8 * 2**20:
            continue
        rows = unpack(file)
        if isinstance(rows, dict):
            rows = rows.get("rows", [])
        if isinstance(rows, list):
            for w in rows:
                if isinstance(w, dict) and isinstance(w.get("source"), str):
                    old_pages.add(w["source"])
    with ZipFile(FILE) as z:
        dev = json.loads(z.read("data/dev.json"))
    used = {w["issuer"] for w in trained}
    eligible = []
    for w in dev:
        issuer = w["filename"].split("/")[0]
        if issuer in used or w["filename"] in old_pages or w["id"] in SOURCE_EXCLUSIONS:
            continue
        a = w["annotation"]
        if len(a["turn_program"]) < 2:
            continue
        p, q = a["turn_program"][:2]
        if "#0" not in q or p not in q or len(p) < 5:
            continue
        turns = []
        try:
            for i, prog in enumerate((p, q)):
                steps, arithmetic = convert(prog)
                if not arithmetic:
                    break
                result = calculate(steps)
                if abs(
                    Fraction(result["value"]) - Fraction(str(a["exe_ans_list"][i]))
                ) > Fraction(51, 10**7):
                    break
                turns.append(
                    dict(
                        question=a["dialogue_break"][i],
                        program=prog,
                        steps=steps,
                        observed=result,
                        reference=a["exe_ans_list"][i],
                    )
                )
        except (ValueError, ArithmeticError, SyntaxError, KeyError, TypeError):
            continue
        if len(turns) != 2:
            continue
        body = "\n".join(
            [
                *w["pre_text"],
                "\n".join(" | ".join(r) for r in w["table"]),
                *w["post_text"],
            ]
        )
        passage = windows(body)
        if passage & excluded:
            continue
        # Prefer completely unused issuers, then independently select a page by hash.
        eligible.append(
            (
                issuer in old_issuers,
                fingerprint([SEED, w["id"]]),
                issuer,
                w,
                body,
                turns,
                passage,
            )
        )
    out = []
    enrolled = set()
    for prior, _, issuer, w, body, turns, passage in sorted(
        eligible, key=lambda x: x[:2]
    ):
        if issuer in enrolled or passage & excluded:
            continue
        enrolled.add(issuer)
        excluded.update(passage)
        out.append(
            dict(
                id=w["id"],
                file=w["filename"],
                cluster=issuer,
                family="report",
                previous_issuer=prior,
                body=body,
                turns=turns,
            )
        )
        if len(out) == 24:
            break
    return out


def build():
    for path in (COQA, TEACHER / "receipt.json", DATA, TARGETS):
        committed(path)
    assert read(TEACHER / "receipt.json")["passed"]
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    episodes = read(COQA)
    rows = []
    panel = []
    spans = set()
    for split in ("train", "dev"):
        source = CACHE / f"coqa-{split}-v1.0.json"
        assert digest(source) == episodes["hashes"][split]
        stories = {r["id"]: r["story"] for r in read(source)["data"]}
        for e in episodes["episodes"][split]:
            story = stories[e["id"]]
            assert fingerprint(story) == e["story"]
            if split == "dev":
                panel.append(
                    dict(
                        id=e["id"],
                        cluster=e["file"],
                        family="reading",
                        body=story,
                        turns=e["turns"],
                    )
                )
                continue
            spans.update(windows(story))
            history = [dict(role="system", content=STYLE)]
            for j, t in enumerate(e["turns"]):
                question = (story + "\n\n" if j == 0 else "") + t["question"]
                history.append(user(question))
                proof = story[slice(*t["span"])]
                assert fingerprint(proof) == t["source_text"]
                gold = t["answers"][0]
                p, y = token_pair(
                    tok,
                    dict(
                        messages=history, tools=[], target=answer(gold, reasoning=proof)
                    ),
                )
                rows.append(
                    dict(
                        id=e["id"] + f"/{j}",
                        group="conversation_read",
                        cluster=e["file"],
                        p=p,
                        y=y,
                    )
                )
                history.append(answer(gold))
    material = {w["id"]: w for w in unpack(DATA)}
    for w in material.values():
        spans.update(windows(w["document"]))
    audit = []
    traces = unpack(TEACHER / "traces.json.gz")
    for r in traces:
        if not r["grade"]["accepted"]:
            continue
        w = material[r["id"]]
        i = r["turn"]
        final, why = REVIEW[w["issuer"]]
        audit.append(
            dict(
                id=r["id"],
                turn=i,
                include=final is not None,
                reason=why,
                delivery=final,
            )
        )
        if final is None:
            continue
        hist, q = inputs(w, i)
        hist[0] = dict(role="system", content=FIN)
        hist.append(user(q))
        t = r["trajectory"]["turns"][0]
        y = t["token_ids"]
        p = tok.encode(
            render(tok, hist, declarations([PROVIDER])), add_special_tokens=False
        )
        assert y[-1] == tok.convert_tokens_to_ids("<|im_end|>")
        rows.append(
            dict(
                id=w["id"] + "/teacher",
                group="teacher_action",
                cluster=w["issuer"],
                p=p,
                y=y,
            )
        )
        steps = t["call"]["arguments"]["steps"]
        calc = t["observation"]
        amount = Fraction(calc["value"])
        import re

        display = Fraction(re.search(r"\d[\d,]*(?:\.\d+)?", final)[0].replace(",", ""))
        assert abs(amount - display) <= Fraction(6, 1000), (
            w["issuer"],
            amount,
            display,
        )
        hist += [call(PROVIDER, steps=steps), observed(calc)]
        p, y = token_pair(
            tok,
            dict(messages=hist, tools=declarations([PROVIDER]), target=answer(final)),
        )
        rows.append(
            dict(
                id=w["id"] + "/delivery",
                group="teacher_delivery",
                cluster=w["issuer"],
                p=p,
                y=y,
            )
        )
    assert len(audit) == len(REVIEW)
    old = unpack(TARGETS)
    for group, quota in sorted(
        dict(calculate=160, quantity=160, reasoning=112, reuse=48, redirect=48).items()
    ):
        bucket = sorted(
            (w for w in old if w["group"] == group),
            key=lambda w: fingerprint([SEED, w["id"]]),
        )[:quota]
        assert len(bucket) == quota
        for w in bucket:
            rows.append(
                dict(
                    id=w["id"],
                    group=group,
                    cluster=w.get("cluster", w["id"]),
                    p=w["p"],
                    y=w["y"],
                )
            )
    other = sorted(
        (
            w
            for w in old
            if w["group"]
            not in ("calculate", "quantity", "reasoning", "reuse", "redirect")
        ),
        key=lambda w: fingerprint([SEED, w["id"]]),
    )[:96]
    for w in other:
        rows.append(
            dict(
                id=w["id"],
                group="retention",
                cluster=w.get("cluster", w["id"]),
                p=w["p"],
                y=w["y"],
            )
        )
    panel += reports(tok, spans)
    for w in panel:
        sys = STYLE if w["family"] == "reading" else FIN
        defs = [] if w["family"] == "reading" else declarations([PROVIDER])
        initial = [
            dict(role="system", content=sys),
            user(w["body"] + "\n\n" + w["turns"][0]["question"]),
        ]
        ids = tok.encode(render(tok, initial, defs), add_special_tokens=False)
        w.update(system=sys, initial=fingerprint(ids), opening=len(ids))
    rows.sort(key=lambda w: fingerprint([SEED, w["id"]]))
    panel.sort(key=lambda w: fingerprint([SEED, w["id"]]))
    assert len({r["id"] for r in rows}) == len(rows)
    ROOT.mkdir(parents=True, exist_ok=True)
    persist_rows(TOKENS, rows)
    persist_rows(PANEL, panel)
    persist(
        ROOT / "teacher-review.json",
        dict(teacher=digest(TEACHER / "receipt.json"), review=audit),
    )
    summary = dict(
        rows=len(rows),
        tokens=sum(len(w["y"]) for w in rows),
        groups=dict(Counter(w["group"] for w in rows)),
        panel=dict(Counter(w["family"] for w in panel)),
        previous_issuer=sum(bool(w.get("previous_issuer")) for w in panel),
        source=digest(TOKENS),
        fresh=digest(PANEL),
        teacher_review=digest(ROOT / "teacher-review.json"),
    )
    persist(ROOT / "inventory.json", summary)
    print(summary)


if __name__ == "__main__":
    build()
