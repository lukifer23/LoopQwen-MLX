"""Command line entry points for local experiments."""

import argparse
import json
from pathlib import Path

from .io import save


def main():
    parser = argparse.ArgumentParser(description="Shared-depth Qwen3.5 on Apple Metal")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("doctor", "Inspect hardware and execute a Metal operation"),
        ("download", "Fetch the pinned official checkpoint"),
        ("baseline", "Measure the original text model"),
        ("cache-check", "Verify cached logits across depths"),
        ("gradient-check", "Verify adapter gradients and optimizer changes"),
        ("gdn-check", "Compare Metal and reference GDN gradients"),
    ):
        sub.add_parser(name, help=help_text)
    p = sub.add_parser("data", help="Build deterministic tokenized pilot splits")
    p.add_argument("--preset", choices=["pilot"], default="pilot")
    p.add_argument(
        "--procedural-only", action="store_true", help="Skip public GSM8K examples"
    )
    p = sub.add_parser(
        "probe",
        help="Measure shared parameter identity, K=1 parity and forward latency",
    )
    p.add_argument("--start-layer", type=int, default=10)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--passes", type=int, choices=[1, 2, 3], default=2)
    p = sub.add_parser("train", help="Train or resume a named experiment")
    p.add_argument("--config", default="configs/smoke.yaml")
    p.add_argument(
        "--until", type=int, help="Stop and checkpoint after this absolute update"
    )
    p = sub.add_parser(
        "benchmark", help="Measure inference, training, GDN, placement and loss"
    )
    p.add_argument("--config", default="configs/benchmark.yaml")
    for name, desc in (
        ("evaluate", "Score four controls, variable depth and the dense budget arm"),
        ("compare", "Display the saved quantitative comparison"),
        ("report", "Write the Markdown experiment report"),
    ):
        p = sub.add_parser(name, help=desc)
        p.add_argument("--experiment", choices=["pilot", "extended"], default="pilot")
    p = sub.add_parser(
        "generate", help="Generate greedily with a new execution-site cache"
    )
    p.add_argument(
        "--checkpoint", required=True, help="Adapter checkpoint or merged export"
    )
    p.add_argument("--prompt", required=True)
    p.add_argument("--passes", type=int, choices=[1, 2, 3])
    p.add_argument("--max-tokens", type=int, default=64)
    p = sub.add_parser(
        "export", help="Merge LoRA and bundle a standalone recurrent loader"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    cmd = args.command
    if cmd == "doctor":
        from .hardware import doctor

        result = doctor()
        save("outputs/doctor.json", result)
    elif cmd == "download":
        from .model_loader import download

        result = str(download())
    elif cmd == "baseline":
        from .baseline import baseline

        result = baseline()
    elif cmd == "data":
        from .data.build_data import build_data

        result = build_data(public=not args.procedural_only)
    elif cmd == "probe":
        from .probe import probe
        from .recurrence import Span

        result = probe(Span(args.start_layer, args.num_layers, args.passes))
    elif cmd == "cache-check":
        from .cache_check import cache_check

        result = cache_check()
    elif cmd == "gradient-check":
        from .training.gradient_check import gradient_check

        result = gradient_check()
    elif cmd == "gdn-check":
        from .benchmark.bench_gdn import bench_gdn, check_projection_gradients

        result = dict(primitive=bench_gdn(), integration=check_projection_gradients())
    elif cmd == "train":
        from .training.trainer import train

        result = train(args.config, args.until)
    elif cmd == "evaluate":
        from .evaluation.comparison import evaluate

        result = evaluate(args.experiment)
    elif cmd in ("report", "compare"):
        from .evaluation.comparison import report

        path = report(args.experiment)
        result = path if cmd == "report" else Path(path).read_text()
    elif cmd == "generate":
        from dataclasses import replace

        from .evaluation.generation_eval import generate

        if (Path(args.checkpoint) / "config.json").exists():
            from .export_runtime import load_export

            model, tok = load_export(args.checkpoint)
        else:
            from .checkpoint import load_trained

            model, tok, _ = load_trained(args.checkpoint)
        if args.passes:
            model.span = replace(model.span, passes=args.passes)
        result = generate(model, tok, args.prompt, args.max_tokens)
    elif cmd == "export":
        from .export import export

        result = export(args.checkpoint, args.output)
    elif cmd == "benchmark":
        from .benchmark.bench_inference import benchmark

        benchmark(args.config)
        result = Path("outputs/benchmark.md").read_text()
    print(result if isinstance(result, str) else json.dumps(result, indent=2))
