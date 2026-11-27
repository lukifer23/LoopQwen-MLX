"""Generate from a user's full input and optionally measure the same request on ordinary Base."""

import gc
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import mlx.core as mx

from . import REVISION
from .evaluation.generation_eval import generate
from .identity import fingerprint
from .worker import acquire


def candidate(args, parser):
    root = Path(args.checkpoint)
    if (root / "refinement.safetensors").is_file():
        from .selective_refinement import load_checkpoint

        if args.passes not in (None, 2) or args.offload_dir is not None:
            parser.error(
                "The selective language checkpoint uses two resident native passes"
            )
        model, tok = load_checkpoint(root)
    elif (root / "config.json").exists():
        from .export_runtime import load_export

        model, tok = load_export(root)
    else:
        from .deployment import load_trained

        model, tok, _ = load_trained(root)
    if getattr(args, "document_reader", None):
        from .document_reader import install

        if args.passes or args.offload_dir is not None:
            parser.error(
                "Document reading uses the released recurrent graph and resident full-history cache"
            )
        install(model, tok, args.document_reader, root)
    if args.passes:
        model.span = replace(model.span, passes=args.passes)
    return model, tok


def render(tok, source, thinking):
    if not thinking:
        return source, ()
    text = tok.apply_chat_template(
        [{"role": "user", "content": source}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    stops = tok.encode("<|im_end|>", add_special_tokens=False)
    if not text.endswith("<think>\n") or len(stops) != 1:
        raise ValueError("Tokenizer does not support the pinned thinking-turn protocol")
    return text, stops


def run(args, parser):
    acquire()
    source = (
        args.prompt_file.read_bytes().decode("utf-8")
        if args.prompt_file
        else args.prompt
    )
    if not args.compare_base:
        model, tok = candidate(args, parser)
        text, stops = render(tok, source, args.thinking)
        return generate(
            model,
            tok,
            text,
            args.max_tokens,
            offload_dir=args.offload_dir,
            stop_ids=stops,
        )

    from .model_loader import ordinary as base

    runs = {}
    policy = None
    for side in ("recurrent", "base"):
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        mx.reset_peak_memory()
        start = perf_counter()
        model, tok = candidate(args, parser) if side == "recurrent" else base()[:2]
        loaded = perf_counter() - start
        text, stops = render(tok, source, args.thinking)
        ids = tok.encode(text, add_special_tokens=False)
        current = (ids, sorted(set(tok.eos_token_ids) | set(stops)))
        if policy is None:
            policy = current
        elif current != policy:
            raise ValueError(
                "Input IDs or generation stop IDs differ across checkpoints"
            )
        first = []
        count = len(ids)

        def observe(consumed, length=count, observed=first, origin=start):
            if consumed == length and not observed:
                observed.append(perf_counter() - origin)

        output = generate(
            model,
            tok,
            text,
            args.max_tokens,
            offload_dir=args.offload_dir,
            stop_ids=stops,
            progress=observe,
        )
        mx.synchronize()
        wall = perf_counter() - start
        if len(first) != 1 or output["prompt_tokens"] != len(ids):
            raise ValueError("Missing or inconsistent first-token measurement")
        output.pop("prompt")
        runs[side] = dict(
            **output,
            initial_sha256=fingerprint(ids),
            load_seconds=loaded,
            cold_first_seconds=first[0],
            request_seconds=wall,
        )
        del model, tok
    return {
        "recurrent_checkpoint": str(args.checkpoint),
        "base_revision": REVISION,
        "order": list(runs),
        "initial_prompt_identical": True,
        "prompt": text,
        "stop_ids": policy[1],
        "runs": runs,
    }
