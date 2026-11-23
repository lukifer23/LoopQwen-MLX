import gc
from collections import Counter
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import yaml

from ..checkpoint import load_trained, location
from ..data.build_data import ROOT, load_split
from ..io import read, save
from ..looped_model import LoopedModel
from ..model_loader import load_base
from ..recurrence import Span
from ..training.trainer import evaluate_rows
from .generation_eval import generate
from .loss_eval import language_loss
from .math_eval import math_eval


def verify_pair(experiment):
    states = [
        read(location(Path("checkpoints") / f"{experiment}_{kind}") / "state.json")
        for kind in ("dense", "loop")
    ]
    a, b = states
    for key in (
        "manifest",
        "step",
        "input_tokens",
        "loss_tokens",
        "order_hash",
        "trainable",
        "cursor",
    ):
        if a[key] != b[key]:
            raise ValueError(f"Unmatched runs: {key}")
    allowed = {"name", "passes"}
    if any(
        a["config"][key] != b["config"][key] for key in a["config"].keys() - allowed
    ):
        raise ValueError("The dense and loop training configurations differ")
    if (a["config"]["passes"], b["config"]["passes"]) != (1, 2) or a["step"] != a[
        "config"
    ]["steps"]:
        raise ValueError("Evaluation requires finished K=1 and K=2 twins")
    if read(ROOT / "manifest.json") != a["manifest"]:
        raise ValueError("Current dataset differs from the training manifest")
    return states


def training_summary(state):
    history = state["history"]
    seconds = sum(r["seconds"] for r in history)
    end = history[-1]
    return dict(
        updates=state["step"],
        inputs=state["input_tokens"],
        labels=state["loss_tokens"],
        final_update_loss=end["loss"],
        final_validation=end["validation"],
        update_s=seconds / len(history),
        tokens_s=state["input_tokens"] / seconds,
        peak_gib=max(r["peak_bytes"] for r in history) / 2**30,
        update_seconds_total=seconds,
        order_hash=state["order_hash"],
    )


def evaluate(experiment):
    states = verify_pair(experiment)
    settings = yaml.safe_load(Path("configs/evaluation.yaml").read_text())
    seen = Counter()
    proc = []
    for row in load_split("test"):
        if seen[row["category"]] < settings["procedural_per_category"]:
            proc.append(row)
            seen[row["category"]] += 1
    gsm = (
        load_split("gsm_test")[: settings["gsm_examples"]]
        if "gsm_test" in states[0]["manifest"]["splits"]
        else []
    )
    out = Path("outputs") / experiment
    suite = {}
    for name, trained, k, matched in (
        ("base", None, 1, False),
        ("untrained_loop", None, 2, False),
        ("trained_dense", "dense", 1, False),
        ("trained_loop", "loop", 2, False),
        ("loop_at_1", "loop", 1, False),
        ("loop_at_3", "loop", 3, False),
        ("dense_budget", "dense", 1, True),
    ):
        if trained:
            model, tok, _ = load_trained(
                Path("checkpoints") / f"{experiment}_{trained}"
            )
        else:
            base, tok, _ = load_base()
            model = LoopedModel(base, Span())
            del base
        model.span = replace(model.span, passes=k)
        model.eval()
        mx.clear_cache()
        mx.reset_peak_memory()
        results = dict(
            k=k,
            prose=language_loss(model, tok),
            answer_loss=evaluate_rows(model, proc),
            procedural=math_eval(model, tok, proc, settings["max_tokens"], matched),
            suite=[
                generate(model, tok, p, settings["suite_tokens"])
                for p in settings["prompts"]
            ],
        )
        if gsm:
            results["gsm"] = math_eval(model, tok, gsm, settings["max_tokens"], matched)
        if gsm and settings.get("math500_numeric_examples", 0):
            from ..data.math500 import numeric_slice

            math_rows = numeric_slice(tok, settings["math500_numeric_examples"])
            results["math500_numeric"] = math_eval(
                model, tok, math_rows, settings["max_tokens"], matched
            )
        save(out / (name + ".json"), results)
        records = results["procedural"]["records"]
        suite[name] = dict(
            k=k,
            math500_numeric=results.get("math500_numeric", {})
            .get("groups", {})
            .get("all"),
            prose_nll=results["prose"]["nll"],
            answer_nll=results["answer_loss"]["nll"],
            perplexity=results["answer_loss"]["perplexity"],
            answer_tokens=results["answer_loss"]["tokens"],
            procedural=results["procedural"]["groups"]["all"],
            gsm=results.get("gsm", {}).get("groups", {}).get("all"),
            seconds=sum(r["elapsed_s"] for r in records),
            layer_executions=sum(r["layer_executions"] for r in records),
        )
        print(name, suite[name], flush=True)
        del model
        gc.collect()
        mx.clear_cache()
    dense_rows = read(out / "trained_dense.json")["procedural"]["records"]
    loop_rows = read(out / "trained_loop.json")["procedural"]["records"]
    pairs = list(zip(dense_rows, loop_rows, strict=True))
    if any(d["id"] != l["id"] for d, l in pairs):
        raise ValueError("Evaluation examples differ")
    paired = dict(
        both=sum(d["correct"] and l["correct"] for d, l in pairs),
        dense_only=sum(d["correct"] and not l["correct"] for d, l in pairs),
        loop_only=sum(l["correct"] and not d["correct"] for d, l in pairs),
        neither=sum(not d["correct"] and not l["correct"] for d, l in pairs),
    )
    summary = dict(
        paired=paired,
        settings=settings,
        manifest=states[0]["manifest"],
        models=suite,
        training={n: training_summary(s) for n, s in zip(("dense", "loop"), states)},
    )
    save(out / "comparison.json", summary)
    return summary


def report(experiment):
    out = Path("outputs") / experiment
    if not (out / "comparison.json").exists():
        raise FileNotFoundError(
            f"Run python -m loopqwen evaluate --experiment {experiment}"
        )
    summary = read(out / "comparison.json")
    lines = [
        f"# {experiment} comparison",
        "",
        "| Model | Answer NLL | Prose NLL | Procedural exact | GSM8K exact | Actual layer-token executions¹ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in summary["models"].items():
        p, g = r["procedural"], r["gsm"]
        gsm = f"{g['correct']}/{g['n']}" if g else "n/a"
        lines.append(
            f"| {name} | {r['answer_nll']:.4f} | {r['prose_nll']:.4f} | {p['correct']}/{p['n']} | {gsm} | {r['layer_executions']:,} |"
        )
    lines += [
        "",
        "¹ Procedural generation only. Includes prefill. Each generated token after the first requires one decode forward. EOS can leave budget unused.",
        "",
        "| Training | Updates | Input tokens | Scored tokens | Final update NLL | Seconds/update | Input tok/s | Peak MLX GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in summary["training"].items():
        lines.append(
            f"| {name} | {r['updates']} | {r['inputs']} | {r['labels']} | {r['final_update_loss']:.4f} | {r['update_s']:.3f} | {r['tokens_s']:.1f} | {r['peak_gib']:.2f} |"
        )
    lines += [
        "",
        "The dense budget arm receives a longer per-prompt output allowance at the same nominal decoder-layer budget. The vocabulary head, different layer types and adapter compute make this an approximate compute control.",
    ]
    path = out / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return str(path)
