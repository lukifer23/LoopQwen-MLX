"""A shared weight-only operating point for the two completed native model fits."""

import argparse
from dataclasses import replace
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import mlx.core as mx
from mlx import nn, optimizers
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

from loopqwen.evaluation.mechanism import unpack
from loopqwen.export_runtime import load_export_tokenizer
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.middle_depth import MiddleAssistant, architecture
from loopqwen.tool_model import quantize_prefix
from scripts.middle_depth_budget import BASE, ROOT, worker
from scripts.middle_depth_fit import loss
from scripts.middle_depth_material import ROWS
from scripts.middle_depth_recover import fit_directory

PLAN = Path("research/diagnostics/middle-depth/weight-quantization.md")
BUNDLE = Path("exports/middle-depth/quantized/base-affine-q4-g64")
WEIGHTS = BUNDLE / "model.safetensors"
PROOF = ROOT / "quantization" / "conversion.json"


def convert():
    if PROOF.is_file():
        verified()
        return
    with worker("profile", "quantized/conversion") as guard:
        tick = monotonic()
        file = BASE / "model.safetensors"
        sha = digest(file)
        if sha != read(BASE / "tool-export-files.json")[file.name]:
            raise ValueError("The certified text donor differs")
        source = mx.load(str(file))
        packed = {}
        names = []
        for name, value in source.items():
            guard()
            if value.ndim == 2:
                if (
                    value.dtype != mx.bfloat16
                    or value.shape[-1] % 64
                    or not name.endswith(".weight")
                ):
                    raise ValueError(f"Unexpected Base matrix {name}")
                weight, scales, biases = mx.quantize(
                    value, group_size=64, bits=4, mode="affine"
                )
                mx.eval(weight, scales, biases)
                prefix = name.removesuffix("weight")
                packed.update(
                    {name: weight, prefix + "scales": scales, prefix + "biases": biases}
                )
                names.append(name)
            else:
                packed[name] = value
        if len(names) != 187 or len(source) != 320:
            raise ValueError("Different Qwen3.5 text architecture")
        BUNDLE.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=BUNDLE) as tmp:
            path = Path(tmp) / WEIGHTS.name
            mx.save_safetensors(str(path), packed)
            guard(force=True)
            reduced = digest(path)
            if WEIGHTS.exists() and digest(WEIGHTS) != reduced:
                raise ValueError("Previously packed backbone changed")
            path.replace(WEIGHTS)
        save(
            PROOF,
            {
                "donor": sha,
                "weights": reduced,
                "plan": digest(PLAN),
                "bytes": WEIGHTS.stat().st_size,
                "bf16_bytes": file.stat().st_size,
                "seconds": monotonic() - tick,
                "peak": mx.get_peak_memory(),
                "matrices": sorted(names),
                "mode": "affine",
                "bits": 4,
                "group_size": 64,
            },
        )


def verified():
    proof = read(PROOF)
    if (
        proof["weights"] != digest(WEIGHTS)
        or proof["plan"] != digest(PLAN)
        or proof["donor"] != read(BASE / "tool-export-files.json")["model.safetensors"]
    ):
        raise ValueError("Invalid quantized donor")
    return proof


def load(policy):
    proof = verified()
    spec = read(policy / "registration.json")
    if spec["donor"] != "base" or spec["weights"] != digest(
        policy / "adapters.safetensors"
    ):
        raise ValueError("Wrong fitted policy for the quantized donor")
    cfg = read(BASE / "config.json")["text_config"]
    text = TextModel(TextModelArgs.from_dict(cfg))
    quantize_prefix(text, len(text.layers), 4)
    text.load_weights(str(WEIGHTS), strict=True)
    model = architecture(text, spec["trained_depth"], spec["seed"])
    saved = mx.load(str(policy / "adapters.safetensors"))
    trainable = dict(tree_flatten(model.trainable_parameters()))
    if trainable.keys() != saved.keys() or any(
        v.shape != saved[k].shape or v.dtype != saved[k].dtype
        for k, v in trainable.items()
    ):
        raise ValueError("Quantization changed the fitted adapter layout")
    model.load_weights(list(saved.items()), strict=False)
    model.span = replace(model.span, passes=spec["serving_depth"])
    model.eval()
    mx.eval(model.parameters())
    tok = load_export_tokenizer(BASE, eos_token_ids=[cfg["eos_token_id"]])
    return model, tok, proof["weights"]


def probe(arm, precision, repeat):
    policy = fit_directory(arm) / "step-0576"
    with worker("profile", f"quantized/{arm}/{precision}/{repeat}") as guard:
        tick = monotonic()
        if precision == "q4":
            model, _, weights = load(policy)
        else:
            runner = MiddleAssistant(BASE, policy)
            model = runner._tool_model
            weights = read(BASE / "tool-export-files.json")["model.safetensors"]
            mx.eval(model.parameters())
        loaded = monotonic()
        resident = mx.get_active_memory()
        rows = unpack(ROWS)
        row = next(r for r in rows if len(r["p"]) >= 160 and 0 < len(r["y"]) <= 60)
        ids = row["p"][:160]
        times = []
        for _ in range(2):
            guard(force=True)
            cache = model.make_cache()
            x = mx.array([ids[:128]])
            begin = monotonic()
            logits = model(x, cache)[:, -1]
            mx.eval(logits)
            opening = monotonic() - begin
            begin = monotonic()
            for token in ids[128:]:
                logits = model(mx.array([[token]]), cache)[:, -1]
                mx.eval(logits)
                guard()
            times.append({"opening": opening, "decode": monotonic() - begin})
            del cache, logits
        report = {
            "arm": arm,
            "physical_layer_visits": len(model.sites),
            "precision": precision,
            "repeat": repeat,
            "backbone": weights,
            "adapter": digest(policy / "adapters.safetensors"),
            "row": row["id"],
            "input": fingerprint(ids),
            "load_seconds": loaded - tick,
            "loaded_bytes": resident,
            "peak_inference": mx.get_peak_memory(),
            "positions": len(ids),
            "generated_positions": 0,
            "forced_decode_positions": 32,
            "timings": times,
        }
        if precision == "q4" and repeat == 0:
            r = next(
                r
                for r in rows
                if 64 <= len(r["p"]) + len(r["y"]) <= 128 and r["negatives"]
            )
            update_start = monotonic()
            mx.reset_peak_memory()
            model.train()
            ids = mx.array([r["p"] + r["y"][:-1]])
            h = model.prefix_hidden(ids)
            mx.eval(h)
            ks = range(1, read(policy / "registration.json")["trained_depth"] + 1)

            def objective(net):
                return sum(
                    loss(net, h, mx.array(r["y"]), len(r["p"]) - 1, k, r["negatives"])
                    for k in ks
                ) / len(ks)

            value, g = nn.value_and_grad(model, objective)(model)
            g, norm = optimizers.clip_grad_norm(g, 1.0)
            before = dict(tree_flatten(model.trainable_parameters()))
            guard(force=True)
            opt = optimizers.AdamW(learning_rate=0.00012)
            opt.update(model, g)
            mx.eval(model.trainable_parameters(), opt.state)
            changed = [
                k
                for k, v in tree_flatten(model.trainable_parameters())
                if not mx.array_equal(v, before[k]).item()
            ]
            post = objective(model).item()
            if (
                not changed
                or not all(map(isfinite, (post, value.item(), norm.item())))
                or norm.item() <= 0
            ):
                raise ValueError("Invalid native quantized adapter update")
            report["disposable_update"] = {
                "source": r["id"],
                "before": value.item(),
                "after": post,
                "gradient": norm.item(),
                "changed": changed,
                "depths": list(ks),
                "input_positions": ids.shape[1],
                "target_positions": len(r["y"]),
                "seconds": monotonic() - update_start,
                "peak_bytes": mx.get_peak_memory(),
            }
        guard(force=True)
        save(ROOT / "quantization" / f"{arm}-{precision}-{repeat}.json", report)


def run():
    from scripts.middle_depth_wave import dispatch, generate

    dispatch("quant", "convert")
    for repeat, modes in enumerate((("bf16", "q4"), ("q4", "bf16"))):
        for arm in ("loop", "ordinary") if repeat == 0 else ("ordinary", "loop"):
            for mode in modes:
                path = ROOT / "quantization" / f"{arm}-{mode}-{repeat}.json"
                if not path.is_file():
                    dispatch(
                        "quant",
                        "probe",
                        "--arm",
                        arm,
                        "--precision",
                        mode,
                        "--repeat",
                        repeat,
                    )
    generate(
        "quantized",
        ["ordinary", "loop"],
        [512],
        {a: fit_directory(a) / "step-0576" for a in ("ordinary", "loop")},
    )


def score():
    from statistics import median

    from scripts.middle_depth_progress import measure, requests

    proof = verified()
    arms = {}
    summaries = {
        k: read(ROOT / "review" / k / "summary.json") for k in ("terminal", "quantized")
    }
    for arm in ("ordinary", "loop"):
        quality = measure("quantized", arm, "terminal", arm, 512)
        outcomes = {s: requests(s, arm, 512) for s in ("terminal", "quantized")}
        followups = {
            world: {
                source: {s: rows[source, world] for s, rows in outcomes.items()}
                for source in ("tiles", "parcels")
            }
            for world in ("own", "shared")
        }
        costs = {}
        for precision in ("bf16", "q4"):
            rows = [
                read(ROOT / "quantization" / f"{arm}-{precision}-{i}.json")
                for i in (0, 1)
            ]
            costs[precision] = {
                "median_seconds_for_32_positions": median(
                    t["decode"] for r in rows for t in r["timings"]
                ),
                "loaded_bytes": max(r["loaded_bytes"] for r in rows),
                "peak": max(r["peak_inference"] for r in rows),
                "identity": [r["adapter"] for r in rows],
                "prefix": [r["input"] for r in rows],
            }
        if (
            costs["q4"]["identity"] != costs["bf16"]["identity"]
            or costs["q4"]["prefix"] != costs["bf16"]["prefix"]
        ):
            raise ValueError("Unmatched fixed-work quantization comparison")
        rates = {
            stage: {
                r["family"]: r["by_eight_seconds"]
                for r in summary
                if r["arm"] == arm and r["cap"] == 512
            }
            for stage, summary in summaries.items()
        }
        assert len(quality) == len(rates["terminal"]) == len(rates["quantized"]) == 3
        retained = all(
            q["losses"] == 0 and rates["quantized"][f] >= rates["terminal"][f]
            for f, q in quality.items()
        )
        arms[arm] = {
            "source_comparison": quality,
            "followups": followups,
            "by_eight_seconds": rates,
            "fixed_work": costs,
            "passes_short_screen": retained
            and all(
                costs["q4"][k] < costs["bf16"][k]
                for k in ("loaded_bytes", "median_seconds_for_32_positions")
            ),
        }
    save(
        ROOT / "quantization" / "short-decision.json",
        {
            "conversion": digest(PROOF),
            "backbone": proof["weights"],
            "arms": arms,
            "both_pass": all(r["passes_short_screen"] for r in arms.values()),
            "scope": "Weight-only short development screen. Separate full-length and cold checks are required for deployment.",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("convert", "probe", "run", "score"))
    p.add_argument("--arm", choices=("ordinary", "loop"))
    p.add_argument("--precision", choices=("q4", "bf16"))
    p.add_argument("--repeat", type=int, choices=(0, 1), default=0)
    a = p.parse_args()
    if a.action == "run":
        run()
    elif a.action == "score":
        score()
    elif a.action == "convert":
        convert()
    else:
        probe(a.arm, a.precision, a.repeat)
