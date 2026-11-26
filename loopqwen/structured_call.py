"""Render an actually chosen semantic action as the first tool turn."""

from time import perf_counter

import mlx.core as mx

from .tool_session import compact


def decoder(action):
    def decode(model, tokenizer, text, max_tokens, **kwargs):
        mx.synchronize()
        at = perf_counter()
        call = action(model, tokenizer, text)
        ids = tokenizer.encode("</think>\n\n" + compact(call), add_special_tokens=False)
        ids.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))
        mx.synchronize()
        elapsed = perf_counter() - at
        if kwargs.get("progress"):
            kwargs["progress"](len(tokenizer.encode(text, add_special_tokens=False)))
        return {
            "token_ids": ids[:max_tokens],
            "elapsed_s": elapsed,
            "ttft_s": elapsed,
            "peak_bytes": mx.get_peak_memory(),
        }

    return decode
