"""Teach role binding after observable wrong-column lookups on genuine annotated tables."""

import csv
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen import qwen_tools
from loopqwen.csv_tools import tools as csv_tools
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from loopqwen.research.table_worlds import canonical
from scripts.program_language_material import answer, call, observed, user
from scripts.provider_recovery_csv import ENROLLED, MULTI, identify
from scripts.provider_recovery_material import ROOT
from scripts.table_policy_material import persist


def variants(header, key, value):
    return {
        "used_answer_field_as_lookup": (value, value),
        "swapped_both_fields": (value, key),
        **{
            "other_lookup_field/" + field: (field, value)
            for field in header
            if field not in (key, value)
        },
    }


def live(folder, w, v, link, mode):
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / "records.csv"
    with file.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(v["rows"])
    (provider,) = csv_tools([file], keyed=True)
    kc, key, _, vc = link
    wrong_k, wrong_v = variants(v["rows"][0], kc, vc)[mode]
    real = {"document": file.name, "key_column": kc, "key": key, "value_column": vc}
    wrong = {**real, "key_column": wrong_k, "value_column": wrong_v}
    try:
        provider.callback(**wrong)
    except ValueError as e:
        if str(e) != "No CSV record matches the key":
            return None
    else:
        return None
    good = provider.callback(**real)
    assert canonical(good["value"]) == canonical(v["answer"])
    msgs = [user(w["question"])]
    # The failed action already contains the correct entity. No wrapper guesses a
    # spelling, repairs an argument, retries a call or chooses the answer at inference.
    msgs.extend([call(provider, **wrong), observed({"error": "ValueError"})])
    yes = call(provider, **real)
    defs = qwen_tools.declarations([provider])
    rows = []
    for group, history, target in (
        ("repair_column_roles", msgs, yes),
        ("answer_after_role_repair", [*msgs, yes, observed(good)], answer(v["answer"])),
    ):
        rows.append(
            {
                "id": w["id"] + "/" + str(v["world"]) + "/" + group,
                "source": w["id"],
                "group": group,
                "messages": history,
                "tools": defs,
                "target": target,
            }
        )
    return rows, {
        "world": v["world"],
        "human": v["answer"],
        "return": fingerprint(good),
        "bad": wrong,
        "correct": real,
    }


def build(output=ROOT):
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    pool = {r["pair"] for r in unpack(ENROLLED) if r["group"] == "operator"}
    rows = []
    meta = []
    excluded = Counter()
    with TemporaryDirectory() as t:
        root = Path(t)
        for w in unpack(MULTI):
            if w["id"] not in pool:
                continue
            links = identify(w, case_change=False)
            if len(links) != 1:
                excluded["unique_source_supported_lookup_absent"] += 1
                continue
            (link,) = links
            kc, key, literal, vc = link
            worlds = w["versions"]
            modes = variants(worlds[0]["rows"][0], kc, vc)
            modes = sorted(modes, key=lambda x: fingerprint([12875, w["id"], x]))
            other = min(
                (
                    v
                    for v in worlds[1:]
                    if canonical(v["answer"]) != canonical(worlds[0]["answer"])
                ),
                key=lambda v: fingerprint([17235, w["id"], v["world"]]),
            )
            enrolled = {worlds[0]["world"], other["world"]}
            complete = False
            for mode in modes:
                records = []
                checked = []
                counts = Counter()
                for v in worlds:
                    result = live(root / w["id"] / str(v["world"]), w, v, link, mode)
                    if not result:
                        break
                    examples, outcome = result
                    checked.append(outcome)
                    if v["world"] in enrolled:
                        for r in examples:
                            p, y = token_pair(tok, r)
                            if len(p) + len(y) > 1792 or len(y) > 384:
                                break
                            counts["opening"] += len(p)
                            counts["labels"] += len(y)
                            records.append(r)
                if len(checked) == len(worlds) and len(records) == 4:
                    # Publish each user question once even when several plausible wrong columns exist.
                    rows.extend(records)
                    meta.append(
                        {
                            "id": w["id"],
                            "context": w["context"],
                            "literal": literal,
                            "key": key,
                            "mode": mode,
                            "enrolled": sorted(enrolled),
                            "versions": checked,
                            **counts,
                        }
                    )
                    complete = True
                    break
            if not complete:
                excluded["no_uniform_live_failure_or_length"] += 1
    assert len(meta) + sum(excluded.values()) == len(pool)
    p = output / "csv-role-training.json.gz"
    if p.is_file():
        assert unpack(p) == rows
    else:
        packed(p, rows)
    d = {
        "training": digest(p),
        "sources": len(meta),
        "turns": len(rows),
        "versions": sum(len(w["versions"]) for w in meta),
        "opening": sum(w["opening"] for w in meta),
        "labels": sum(w["labels"] for w in meta),
        "failure_modes": Counter(w["mode"].split("/")[0] for w in meta),
        "caller_spelling_exact": sum(w["key"] == w["literal"] for w in meta),
        "excluded": excluded,
        "cases": meta,
    }
    persist(output / "csv-role-qualification.json", d)
    print({k: v for k, v in d.items() if k != "cases"})


if __name__ == "__main__":
    build()
