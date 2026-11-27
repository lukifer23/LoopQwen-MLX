"""Measure one shared recurrent instruction path on changed human meanings and live CSV work."""

import argparse
import gc
import re
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from ..calculator_tool import tool
from ..csv_tools import tools as csv_tools
from ..evaluation.answers import NUMBER, numeric
from ..evaluation.mechanism import packed, unpack
from ..interpolated_instruction import InterpolatedAssistant
from ..io import digest, read, save
from ..model_loader import local_snapshot
from ..tool_compare import DenseAssistant
from .core import Budget
from .native_function_study import evaluate, executions, replay, summarize
from .observation_review_study import paired
from .registered import registered
from .report_actions import operand_check


def plain(s):
    return (
        re.sub(r"\s+", " ", s.casefold().replace("’", "'").replace("**", ""))
        .strip(' .!"“”\n')
        .removeprefix("the ")
    )


def referent(response, options):
    s = plain(response)
    names = sorted(map(plain, options), key=len, reverse=True)
    if s in names:
        return s
    pattern = r"(?<!\w)(?:" + "|".join(map(re.escape, names)) + r")(?!\w)"
    explicit = re.findall(
        r"(?:refers? to|referent is|answer is|answer:)\s+(?:the )?(" + pattern + ")", s
    )
    if len(set(explicit)) == 1 and not re.search(r"(?:not|n't)\s+refer", s):
        return explicit[0]
    # Only a single distinct option may occur outside an explicit conclusion.
    found = set(re.findall(pattern, s))
    if len(found) == 1 and not re.search(r"\bnot\b|n't", s):
        return found.pop()
    return None


def score(row, i, tr):
    answer = tr["answer"] or ""
    done = tr["status"] == "answered"
    if row["family"] == "reference":
        decoded = referent(answer, row["options"])
        ok = decoded == plain(row["gold"])
        return {
            "success": bool(done and ok),
            "completed": done,
            "correct_answer": ok,
            "decoded": decoded,
        }
    key = row["keys"][i]
    value = row["values"][i]
    for name in [row["document"], *row["keys"]]:
        answer = answer.replace(name, "")
    gold = Fraction(row["gold"][i])
    v = numeric(answer)
    if v is None:
        nums = re.findall(NUMBER, answer)
        v = numeric(nums[-1]) if nums else None
    ok = v is not None and Fraction(v) == gold
    events = executions(tr)
    lookup = any(
        e["call"]
        == {
            "name": "csv_lookup",
            "arguments": {
                "document": row["document"],
                "key_column": "filename",
                "key": key,
                "value_column": "bytes",
            },
        }
        and e["observation"] == {"value": str(value)}
        for e in events
    )
    calc = [e for e in events if e["call"]["name"] == "calculate"]
    expr = f"{row['values'][1]} - {row['values'][0]}"
    computed = bool(calc and calc[-1]["observation"].get("value") == str(gold))
    bound = bool(i and computed and operand_check(expr, calc))
    return {
        "success": bool(done and ok and lookup and (i == 0 or computed)),
        "correct_answer": ok,
        "completed": done,
        "lookup": lookup,
        "calculated": computed,
        "bound": bound,
    }


def outcome(panel, runs):
    met = {}
    solved = {}
    for side, arm in runs.items():
        groups = {}
        sets = {}
        by = defaultdict(list)
        for w, r in zip(panel["rows"], arm, strict=True):
            assert r["id"] == w["id"]
            by[w["family"], w["cluster"]].extend(
                q["score"]["success"] for q in r["requests"]
            )
        for k in ("all", "reference", "inventory"):
            groups[k] = summarize([r for r in arm if k == "all" or r["family"] == k])
            sets[k] = {
                code
                for code, qs in by.items()
                if (k == "all" or code[0] == k) and len(qs) == 2 and all(qs)
            }
        q = [q for r in arm for q in r["requests"]]
        turns = [t for req in q for t in req["trajectory"]["turns"]]
        groups["rates"] = {
            "opening": sum(len(z["trajectory"]["turns"][0]["prompt_ids"]) for z in q)
            / groups["all"]["first_s"],
            "decode": sum(len(t["token_ids"]) - 1 for t in turns)
            / sum(t["seconds"] - t["ttft_s"] for t in turns),
        }
        groups["bound_differences"] = sum(
            z["score"].get("bound", False) and z["score"]["success"] for z in q
        )
        met[side] = groups
        solved[side] = sets
    a, b = met["candidate"], met["control"]
    ps = {}
    for k in ("all", "reference", "inventory"):
        x, y = solved["candidate"][k], solved["control"][k]
        ps[k] = dict(loop=len(x), base=len(y), **paired(x, y))

    def gain(k):
        return ps[k]["loop"] - ps[k]["base"]

    gates = {
        "meanings": gain("reference") >= 3 and ps["reference"]["p"] <= 0.05,
        "tools": gain("inventory") > 0
        and a["bound_differences"] >= b["bound_differences"],
        "individual": all(
            a[k]["answers"] >= b[k]["answers"]
            and a[k]["completed"] >= b[k]["completed"]
            for k in ("reference", "inventory")
        ),
        "pooled": gain("all") > 0 and ps["all"]["p"] <= 0.05,
        "first": a["all"]["first_s"] < b["all"]["first_s"],
        "wall": a["all"]["wall_s"] < b["all"]["wall_s"],
        "peak": a["all"]["peak"] < b["all"]["peak"],
        "rates": all(a["rates"][k] > b["rates"][k] for k in ("opening", "decode")),
    }
    return {"passed": all(gates.values()), "gates": gates, "pairs": ps, "groups": met}


def write_documents(panel, folder):
    for file, body in panel["documents"].items():
        assert Path(file).name == file
        (Path(folder) / file).write_bytes(body.encode())


def funcs(w, folder):
    return (
        []
        if w["family"] == "reference"
        else csv_tools([Path(folder) / w["document"]], keyed=True) + [tool()]
    )


def verify(root, d):
    spec = read(root / "registration.json")
    assert spec == registered(spec["id"])
    panel = read(spec["material"])
    tok = load_tokenizer(local_snapshot())
    runs = {}
    with TemporaryDirectory() as tmp:
        write_documents(panel, tmp)
        for side in ("candidate", "control"):
            arm = runs[side] = unpack(root / (side + ".json.gz"))
            for w, r in zip(panel["rows"], arm, strict=True):
                assert (
                    w["id"] == r["id"]
                    and w["initial_ids"]
                    == r["requests"][0]["trajectory"]["turns"][0]["prompt_ids"]
                )
                history = []
                for i, (s, q) in enumerate(
                    zip(w["questions"], r["requests"], strict=True)
                ):
                    history.append({"role": "user", "content": s})
                    replay(tok, w, funcs(w, tmp), history, q, spec, side)
                    assert score(w, i, q["trajectory"]) == q["score"]
    assert outcome(panel, runs) == d


def run(name):
    spec = registered(name)
    release = Path(spec["requires"])
    panel = read(spec["material"])
    assert (
        read(release / "decision.json")["passed"]
        and digest(release / "checksums.json") == spec["prerequisite_sha256"]
    )
    root = Path("outputs/research") / name
    root.mkdir(parents=True, exist_ok=True)
    save(root / "registration.json", spec)

    def control():
        base = DenseAssistant(True, "qwen", spec["checkpoint"])
        base.native_plain = True
        return base

    budget = Budget(name, spec["limits"])
    try:
        with TemporaryDirectory() as tmp:
            write_documents(panel, tmp)
            runs = {}
            for side in ("candidate", "control"):
                gc.collect()
                mx.clear_cache()
                path = root / (side + ".json.gz")
                if not path.exists():
                    factory = (
                        (lambda: InterpolatedAssistant(spec["checkpoint"], root))
                        if side == "candidate"
                        else control
                    )
                    packed(
                        path,
                        evaluate(
                            spec,
                            panel,
                            side,
                            budget,
                            factory,
                            lambda w: funcs(w, tmp),
                            scorer=score,
                        ),
                    )
                runs[side] = unpack(path)
            save(root / "decision.json", outcome(panel, runs))
    finally:
        budget.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    run(p.parse_args().name)
