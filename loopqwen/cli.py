"""Command line entry points for local experiments."""

import argparse
import json
from pathlib import Path

from .io import save


def main():
    parser = argparse.ArgumentParser(description="Shared-depth Qwen3.5 on Apple Metal")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        (
            "resources",
            "Inspect project processes, GPU lease, available memory and pressure",
        ),
        ("doctor", "Inspect hardware and execute a Metal operation"),
        ("download", "Fetch the pinned official checkpoint"),
        ("baseline", "Measure the original text model"),
        ("cache-check", "Verify cached logits across depths"),
        ("gradient-check", "Verify adapter gradients and optimizer changes"),
        ("gdn-check", "Compare Metal and reference GDN gradients"),
    ):
        cmd_parser = sub.add_parser(name, help=help_text)
        if name == "cache-check":
            cmd_parser.add_argument(
                "--fp32-diagnostic",
                action="store_true",
                help="Repeat the long K=3 case in FP32",
            )
    p = sub.add_parser("data", help="Build deterministic tokenized pilot splits")
    p.add_argument("--preset", choices=["pilot", "v2", "v3", "v4"], default="pilot")
    p.add_argument("--count", type=int, default=24000, help="V2 training instances")
    p.add_argument("--seed", type=int, default=2602, help="V2 generator seed")
    p.add_argument(
        "--procedural-only", action="store_true", help="Skip public GSM8K examples"
    )
    p.add_argument(
        "--gsm-traces",
        action="store_true",
        help="Train on short published GSM8K solutions",
    )
    p.add_argument(
        "--retention-file",
        help="JSONL general-language documents with text, source and license",
    )
    p = sub.add_parser(
        "probe",
        help="Measure shared parameter identity, K=1 parity and forward latency",
    )
    p.add_argument("--start-layer", type=int, default=10)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--passes", type=int, choices=[1, 2, 3], default=2)
    p = sub.add_parser(
        "fork", help="Start a new strict run from an immutable checkpoint"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--mode", choices=["extend", "weights", "compatible"], required=True)
    p.add_argument("--migration-reason")
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
        p.add_argument("--experiment", default="pilot")
    from .tool_cli import add_arguments

    for name, help_text in (
        ("tools", "Ask the recurrent assistant to use local tools"),
        ("ask", "Ask the offline assistant a question with optional local tools"),
    ):
        p = sub.add_parser(name, help=help_text)
        add_arguments(p)
        p.set_defaults(allow_plain=name == "ask")
        p.add_argument(
            "--compare-base",
            action="store_true",
            help="Repeat each read-only request on untouched Base and report both answers, calls, times and peak bytes",
        )
    p = sub.add_parser(
        "generate", help="Generate greedily with a new execution-site cache"
    )
    p.add_argument(
        "--checkpoint", required=True, help="Adapter checkpoint or merged export"
    )
    prompts = p.add_mutually_exclusive_group(required=True)
    p.add_argument(
        "--offload-dir",
        type=Path,
        help="Use an existing scratch directory for exact BF16 full-history inference",
    )
    p.add_argument(
        "--thinking",
        action="store_true",
        help="Render the input as a thinking-enabled user turn and stop at its first chat boundary",
    )
    prompts.add_argument("--prompt")
    prompts.add_argument(
        "--prompt-file", type=Path, help="Read the complete UTF-8 prompt from disk"
    )
    p.add_argument("--passes", type=int, choices=[1, 2, 3])
    p.add_argument("--max-tokens", type=int, default=64)
    p = sub.add_parser(
        "export", help="Merge LoRA and bundle a standalone recurrent loader"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fusion", choices=["stock", "fp32"], default="stock")
    args = parser.parse_args()
    cmd = args.command
    if cmd == "resources":
        from .worker import resources

        result = resources()
    elif cmd == "doctor":
        from .hardware import doctor

        result = doctor()
        save("outputs/doctor.json", result)
    elif cmd == "download":
        from .model_loader import download

        result = str(download())
    elif cmd == "baseline":
        from .baseline import baseline

        result = baseline()
    elif cmd == "data" and args.preset == "v4":
        from .data.v4 import build

        result = build()
    elif cmd == "data" and args.preset == "v3":
        from .data.v3 import build

        result = build()
    elif cmd == "data" and args.preset == "v2":
        from .data.v2 import build

        result = build(args.count, args.seed)
    elif cmd == "data":
        from .data.build_data import build_data

        result = build_data(
            public=not args.procedural_only,
            gsm_traces=args.gsm_traces,
            retention_file=args.retention_file,
        )
    elif cmd == "probe":
        from .probe import probe
        from .recurrence import Span

        result = probe(Span(args.start_layer, args.num_layers, args.passes))
    elif cmd == "cache-check":
        from .cache_check import cache_check, fp32_diagnostic

        result = fp32_diagnostic() if args.fp32_diagnostic else cache_check()
    elif cmd == "gradient-check":
        from .training.gradient_check import gradient_check

        result = gradient_check()
    elif cmd == "gdn-check":
        from .benchmark.bench_gdn import (
            bench_gdn,
            check_projection_gradients,
            masked_contract,
        )

        result = dict(
            primitive=bench_gdn(),
            integration=check_projection_gradients(),
            masks=masked_contract(),
        )
    elif cmd == "fork":
        from .training.continuation import fork

        result = fork(args.checkpoint, args.config, args.mode, args.migration_reason)
    elif cmd == "train":
        from .training.trainer import train

        result = train(args.config, args.until)
    elif cmd in ("evaluate", "report", "compare") and args.experiment == "phase4":
        if cmd == "evaluate":
            from .evaluation.phase4 import evaluate

            evaluate(109)
            result = "outputs/phase4/evaluation-109.json"
        elif cmd == "compare":
            from .io import read

            index = Path("outputs/phase4/evaluation-109.json")
            archive = Path("docs/measurements/phase4/primary-pairs.json")
            if index.is_file():
                result = read(index)["pairs"]
            elif archive.is_file():
                result = read(archive)
            else:
                raise ValueError("Complete scripts/run_phase4.sh evaluate first")
        else:
            result = Path("docs/PHASE4_RESULTS.md")
            if not result.is_file():
                raise ValueError(
                    "Render the study with python scripts/phase4_report.py"
                )
    elif cmd in ("evaluate", "report", "compare") and args.experiment == "phase3_dev":
        from .io import read
        from .evaluation.phase3_report import report

        if cmd == "evaluate":
            from .evaluation.phase3 import evaluate

            evaluate("main", read("outputs/phase3/selection.json")["selected"])
        path = report()
        result = Path(path).read_text() if cmd == "compare" else path
    elif cmd in ("evaluate", "report", "compare") and args.experiment not in (
        "pilot",
        "extended",
    ):
        from .evaluation.phase2 import evaluate, report

        path = (evaluate if cmd == "evaluate" else report)(args.experiment)
        result = Path(path).read_text() if cmd == "compare" else path
    elif cmd == "evaluate":
        from .evaluation.comparison import evaluate

        result = evaluate(args.experiment)
    elif cmd in ("report", "compare"):
        from .evaluation.comparison import report

        path = report(args.experiment)
        result = path if cmd == "report" else Path(path).read_text()
    elif cmd in ("ask", "tools"):
        from .tool_cli import respond

        result = respond(args, parser)
    elif cmd == "generate":
        from .worker import acquire

        acquire()
        from dataclasses import replace

        from .evaluation.generation_eval import generate

        if (Path(args.checkpoint) / "refinement.safetensors").is_file():
            from .selective_refinement import load_checkpoint

            if args.passes not in (None, 2) or args.offload_dir is not None:
                parser.error(
                    "The selective language checkpoint uses two resident native passes"
                )
            model, tok = load_checkpoint(args.checkpoint)
        elif (Path(args.checkpoint) / "config.json").exists():
            from .export_runtime import load_export

            model, tok = load_export(args.checkpoint)
        else:
            from .deployment import load_trained

            model, tok, _ = load_trained(args.checkpoint)
        if args.passes:
            model.span = replace(model.span, passes=args.passes)
        prompt = (
            args.prompt_file.read_bytes().decode("utf-8")
            if args.prompt_file
            else args.prompt
        )
        stop_ids = ()
        if args.thinking:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            stop_ids = tok.encode("<|im_end|>", add_special_tokens=False)
            if not prompt.endswith("<think>\n") or len(stop_ids) != 1:
                raise ValueError(
                    "Tokenizer does not support the pinned thinking-turn protocol"
                )
        result = generate(
            model,
            tok,
            prompt,
            args.max_tokens,
            offload_dir=args.offload_dir,
            stop_ids=stop_ids,
        )
    elif cmd == "export":
        from .export import export

        result = export(args.checkpoint, args.output, args.fusion)
    elif cmd == "benchmark":
        from .benchmark.bench_inference import benchmark

        benchmark(args.config)
        result = Path("outputs/benchmark.md").read_text()
    print(result if isinstance(result, str) else json.dumps(result, indent=2))
