"""One direct model pair on real open-ended questions in both natural source views."""

import gc
import re
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx

from ..cross_view import CrossViewAssistant, compile_layer
from ..csv_tools import tools
from ..evaluation.extractive import normalized, score
from ..evaluation.mechanism import packed, unpack
from ..evaluation.statistics import exact_mcnemar
from ..export_runtime import load_export_tokenizer
from ..io import digest, read, save
from ..tool_compare import DenseAssistant
from ..tool_model import bind
from .core import Budget
from .cross_view_fit import fit, teacher
from .native_function_study import evaluate, executions, fit_messages, replay, summarize
from .registered import registered

NAME = "cross-view-reading-01"


def providers(w, directory):
    return tools([directory / w["document"]], keyed=True) if w["document"] else []


def write_documents(panel, directory):
    for key, raw in panel["documents"].items():
        path = directory / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw.encode())


def grade(w, i, tr):
    assert i == 0
    done = tr["status"] == "answered"
    text = tr["answer"] if done else ""
    gold = w["gold"]

    def contains(value):
        target = normalized(value)
        return bool(
            target
            and re.search(r"(?<!\w)" + re.escape(target) + r"(?!\w)", normalized(text))
        )

    ok = bool(done and contains(gold) and not (w["other"] and contains(w["other"])))
    exact, f1 = score(text, [gold])
    observed = any(
        e["call"] in w["calls"] and e["observation"] == w["observation"]
        for e in executions(tr)
    )
    return {
        "completed": done,
        "exact": exact,
        "f1": f1,
        "observed": observed,
        "correct_answer": ok,
        "success": ok and (w["family"] == "inline" or observed),
    }


def outcome(panel, runs):
    results = {}
    solved = {}
    for arm, records in runs.items():
        assert [r["id"] for r in records] == [w["id"] for w in panel["rows"]]
        result = {}
        articles = defaultdict(lambda: defaultdict(list))
        for family in ("all", "file", "inline", "retention"):
            items = [r for r in records if family == "all" or r["family"] == family]
            qs = [r["requests"][0] for r in items]
            result[family] = summarize(items)
            result[family].update(
                exact=sum(q["score"]["exact"] for q in qs),
                mean_f1=sum(q["score"]["f1"] for q in qs) / len(qs),
                right_reads=sum(q["score"]["observed"] for q in qs),
            )
        for row, r in zip(panel["rows"], records, strict=True):
            if row["family"] != "retention":
                articles[row["cluster"]][row["family"]].append(
                    r["requests"][0]["score"]["success"]
                )
        assert all(len(v["file"]) == len(v["inline"]) == 2 for v in articles.values())
        solved[arm] = {
            k for k, v in articles.items() if all(v["file"]) and all(v["inline"])
        }
        result["whole_articles"] = {
            mode: sum(all(v[mode]) for v in articles.values())
            for mode in ("file", "inline")
        }
        result["both_modes_both_questions"] = len(solved[arm])
        ts = [t for r in records for t in r["requests"][0]["trajectory"]["turns"]]
        result["opening_tps"] = (
            sum(len(w["initial_ids"]) for w in panel["rows"]) / result["all"]["first_s"]
        )
        result["decode_tps"] = sum(len(t["token_ids"]) - 1 for t in ts) / sum(
            t["seconds"] - t["ttft_s"] for t in ts
        )
        results[arm] = result
    a, b = results["candidate"], results["control"]
    x, y = solved["candidate"], solved["control"]
    wins, losses = len(x - y), len(y - x)
    p = exact_mcnemar(wins, losses)
    gates = {
        "article_gain": wins - losses >= 5 and p <= 0.05,
        "retention": a["retention"]["grounded"] >= b["retention"]["grounded"],
        "completed": a["all"]["completed"] >= b["all"]["completed"],
    }
    for f in ("file", "inline"):
        for k in ("grounded", "exact", "mean_f1"):
            gates[f + "_" + k] = a[f][k] > b[f][k]
    for k in ("wall_s", "first_s", "peak"):
        gates[k] = a["all"][k] < b["all"][k]
    for k in ("opening_tps", "decode_tps"):
        gates[k] = a[k] > b[k]
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "groups": results,
        "wins": wins,
        "losses": losses,
        "p": p,
    }


def tokenizer(s):
    directory = Path(s["checkpoint"])
    return load_export_tokenizer(
        directory,
        eos_token_ids=[read(directory / "config.json")["text_config"]["eos_token_id"]],
    )


def verify(root, d):
    s = registered(NAME)
    assert read(root / "registration.json") == s
    if d.get("status") == "interrupted":
        assert not d["passed"] and d["reason"]
        return
    ts = read(root / "teacher-summary.json")
    assert (
        digest(root / "teacher.json.gz") == ts["targets_sha256"]
        and ts["teacher_weights"] == s["teacher"]["weights_sha256"]
        and ts["first_gold_top_k_rate"] >= s["min_teacher_coverage"]
    )
    fit = read(root / "fit.json")
    assert (
        fit["training"] == digest(Path(s["training"]))
        and fit["weights"] == digest(root / "tool-adapters.safetensors")
        and fit["updates"] == s["updates"]
    )
    cp = read(root / "compiled.json")
    assert (
        len(cp["projections"]) == 7
        and cp["adapter"] == fit["weights"]
        and cp["weights"] == digest(root / "cross-view.safetensors")
    )
    p = read(s["material"])
    t = tokenizer(s)
    runs = {a: unpack(root / (a + ".json.gz")) for a in ("candidate", "control")}
    with TemporaryDirectory() as temp:
        folder = Path(temp)
        write_documents(p, folder)
        for side, records in runs.items():
            for w, r in zip(p["rows"], records, strict=True):
                request = r["requests"][0]
                history = [{"role": "user", "content": w["questions"][0]}]
                replay(t, w, providers(w, folder), history, request, s, side)
                assert (
                    request["trajectory"]["turns"][0]["prompt_ids"] == w["initial_ids"]
                )
                assert request["score"] == grade(w, 0, request["trajectory"])
    assert d == outcome(p, runs)


def run():
    s = registered(NAME)
    prereq = Path(s["requires"])
    assert (
        read(prereq / "decision.json")["passed"]
        and digest(prereq / "checksums.json") == s["prerequisite_sha256"]
    )
    root = Path("outputs/research") / NAME
    root.mkdir(parents=True, exist_ok=True)
    if (root / "decision.json").exists():
        raise FileExistsError("The registered comparison already has a decision")
    if (root / "registration.json").exists():
        assert read(root / "registration.json") == s
    else:
        save(root / "registration.json", s)
    p = read(s["material"])
    budget = Budget(NAME, s["limits"])
    try:
        tok = tokenizer(s)
        teacher(s, tok, root, budget)
        if not (root / "fit.json").exists():
            model = bind(None, Path(s["checkpoint"]) / "tool")
            fit_messages(model, tok, s, root, budget, trainer=fit)
            del model
            gc.collect()
            mx.clear_cache()
        if not (root / "compiled.json").exists():
            compile_layer(s["checkpoint"], root)
        gc.collect()
        mx.clear_cache()
        budget.check()

        def control():
            model = DenseAssistant(True, "qwen", s["checkpoint"])
            model.native_plain = True
            return model

        runs = {}
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_documents(p, folder)
            for side in ("candidate", "control"):
                path = root / (side + ".json.gz")
                gc.collect()
                mx.clear_cache()
                if path.exists():
                    runs[side] = unpack(path)
                else:
                    build = (
                        (lambda: CrossViewAssistant(s["checkpoint"], root))
                        if side == "candidate"
                        else control
                    )
                    runs[side] = evaluate(
                        s,
                        p,
                        side,
                        budget,
                        build,
                        lambda w: providers(w, folder),
                        scorer=grade,
                    )
                    packed(path, runs[side])
        d = outcome(p, runs)
    except (MemoryError, TimeoutError, InterruptedError) as e:
        d = {"status": "interrupted", "passed": False, "reason": str(e)}
    finally:
        budget.close()
    save(root / "decision.json", d)
    print("Closed direct comparison", flush=True)


if __name__ == "__main__":
    run()
