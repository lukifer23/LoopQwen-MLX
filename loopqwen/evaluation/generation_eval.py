from contextlib import nullcontext
from tempfile import TemporaryDirectory
from time import perf_counter

import mlx.core as mx


def generate(
    model,
    tokenizer,
    prompt,
    max_tokens=24,
    *,
    progress=None,
    stop_ids=(),
    offload_dir=None,
):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if (
        not ids
        or not 1 <= max_tokens <= 4096
        or len(ids) + max_tokens > model.args.max_position_embeddings
    ):
        raise ValueError(
            "Supply a nonempty prompt within the context window and 1..4096 output tokens."
        )
    workspace = (
        TemporaryDirectory(dir=offload_dir)
        if offload_dir is not None
        else nullcontext()
    )
    with workspace as directory:
        disk = None
        cache = None
        if directory is None:
            cache = model.make_cache()
        else:
            from ..offloaded_prefill import OffloadedPrefill

            disk = OffloadedPrefill(model, directory)
        inputs = mx.array([ids])
        model.eval()
        mx.synchronize()
        start = perf_counter()
        # Bound prefill activation memory. Each chunk advances every site once.
        if progress is not None:
            progress(0)
        while disk is None and inputs.shape[1] > 128:
            chunk = inputs[:, :128]
            if hasattr(model, "hidden"):
                mx.eval(model.hidden(chunk, cache))
            else:
                mx.eval(model(chunk, cache=cache)[:, -1])
            inputs = inputs[:, 128:]
            if progress is not None:
                progress(len(ids) - inputs.shape[1])
        output = []
        for step in range(max_tokens):
            if disk is not None:
                logits = (
                    disk.prefill(ids, progress) if step == 0 else disk.decode(inputs)
                )
            elif hasattr(model, "hidden"):
                logits = model.project(model.hidden(inputs, cache)[:, -1:])
            else:
                logits = model(inputs, cache=cache)
            token = mx.argmax(logits[:, -1], -1).item()
            if step == 0:
                first = perf_counter()
            output.append(token)
            if progress is not None:
                progress(len(ids) + step)
            if token in tokenizer.eos_token_ids or token in stop_ids:
                break
            inputs = mx.array([[token]])
        end = perf_counter()
    decode = end - first
    result = {
        "prompt": prompt,
        "text": tokenizer.decode(output),
        "token_ids": output,
        "prompt_tokens": len(ids),
        "greedy": True,
        "seed": 0,
        "max_tokens": max_tokens,
        "ttft_s": first - start,
        "elapsed_s": end - start,
        "prefill_tps": len(ids) / (first - start),
        "decode_tps": (len(output) - 1) / decode if len(output) > 1 else None,
        "peak_bytes": mx.get_peak_memory(),
        "active_bytes": mx.get_active_memory(),
    }
    if disk is not None:
        result["scratch_bytes"] = disk.scratch_bytes
    return result
