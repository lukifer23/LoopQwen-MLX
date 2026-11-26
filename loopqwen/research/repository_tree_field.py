"""Grade a read-only whole-repository lookup using the actual Git object sizes."""

import re

from ..evaluation.answers import numeric
from ..evaluation.language7 import interpret


def grade(tokenizer, ids, gold, other):
    ends = {*tokenizer.eos_token_ids, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    parsed = interpret(
        ids,
        tokenizer.convert_tokens_to_ids("<think>"),
        tokenizer.convert_tokens_to_ids("</think>"),
        ends,
        tokenizer,
    )
    answer = parsed["visible"] if ids[-1] in ends else ""
    value = numeric(re.sub(r"\s+bytes[.!]?\s*$", "", answer, flags=re.I))
    numbers = {n.replace(",", "") for n in re.findall(r"(?<!\d)\d[\d,]*(?!\d)", answer)}
    return {
        "visible": answer,
        "exact": value == gold,
        "content": gold in numbers and other not in numbers,
    }


def qualify(result, data, material):
    summary = {}
    for side, rows in data.items():
        first = sum(r["ttft_s"] for r in rows)
        subsequent = sum(len(r["token_ids"]) - 1 for r in rows)
        decoding = sum(r["elapsed_s"] - r["ttft_s"] for r in rows)
        assert decoding > 0 and subsequent > 0
        summary[side] = {
            "cold_seconds": sum(r["cold_seconds"] for r in rows),
            "cold_first": sum(r["cold_first_seconds"] for r in rows),
            "prefill_tps": sum(r["prompt_tokens"] for r in rows) / first,
            "decode_tps": subsequent / decoding,
        }
    evidence = all(
        r.get("context_route") == "register"
        and r["attention_memory"]["start"]
        <= w["gold_start"]
        < w["gold_stop"]
        <= r["attention_memory"]["stop"]
        and all(
            v["reader_biased_queries"]
            == (len(r["token_ids"]) if v["layer"] >= 8 else 0)
            for v in r["page_sites"]
        )
        for r, w in zip(data["candidate"], material["panel"], strict=True)
    )
    a, b = summary["candidate"], summary["control"]
    return {
        **result,
        "passed": bool(
            result["passed"]
            and evidence
            and all(a[k] < b[k] for k in ("cold_first", "cold_seconds"))
            and all(a[k] > b[k] for k in ("prefill_tps", "decode_tps"))
            and result["metrics"]["candidate"]["peak"]
            < result["metrics"]["control"]["peak"]
        ),
        "tree_read_evidence": evidence,
        "resources": summary,
    }
