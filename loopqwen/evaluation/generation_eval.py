from time import perf_counter

import mlx.core as mx


def generate(model, tokenizer, prompt, max_tokens=24):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if (
        not ids
        or not 1 <= max_tokens <= 4096
        or len(ids) + max_tokens > model.args.max_position_embeddings
    ):
        raise ValueError(
            "Supply a nonempty prompt within the context window and 1..4096 output tokens."
        )
    cache = model.make_cache()
    inputs = mx.array([ids])
    model.eval()
    mx.synchronize()
    start = perf_counter()
    output = []
    for step in range(max_tokens):
        logits = model(inputs, cache=cache)
        token = mx.argmax(logits[:, -1], -1).item()
        if step == 0:
            first = perf_counter()
        output.append(token)
        if token in tokenizer.eos_token_ids:
            break
        inputs = mx.array([[token]])
    end = perf_counter()
    decode = end - first
    return dict(
        prompt=prompt,
        text=tokenizer.decode(output),
        token_ids=output,
        prompt_tokens=len(ids),
        greedy=True,
        seed=0,
        max_tokens=max_tokens,
        ttft_s=first - start,
        elapsed_s=end - start,
        prefill_tps=len(ids) / (first - start),
        decode_tps=(len(output) - 1) / decode if len(output) > 1 else None,
        peak_bytes=mx.get_peak_memory(),
        active_bytes=mx.get_active_memory(),
    )
