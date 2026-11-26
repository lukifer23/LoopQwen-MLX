"""Greedy first-turn reasoning from the shared ordinary decoder followed by a recurrent action."""

from time import perf_counter

import mlx.core as mx


def prefill(model, ids, progress):
    cache = model.make_cache()
    for start in range(0, len(ids), 128):
        block = ids[start : start + 128]
        hidden = model.hidden(mx.array([block]), cache)
        mx.eval(hidden)
        if progress:
            progress(start + len(block))
    return cache, hidden[:, -1:]


def draft(ordinary, ids, limit, boundaries, progress=None):
    """Emit ordinary reasoning until its first proposed turn boundary or the fixed allowance."""
    if not limit:
        return [], None, None
    cache, hidden = prefill(ordinary, ids, progress)
    tokens = []
    first = None
    last_proposal = None
    for step in range(limit):
        token = int(ordinary.project(hidden).reshape(-1).argmax().item())
        if token in boundaries:
            last_proposal = token
            break
        if first is None:
            first = perf_counter()
        tokens.append(token)
        if progress:
            progress(len(ids) + len(tokens))
        if step + 1 < limit:
            hidden = ordinary.hidden(mx.array([[token]]), cache)[:, -1:]
    return tokens, last_proposal, first


class DraftAction:
    """Per-request decoder. Every actually emitted draft token counts against the common cap."""

    def __init__(self, ordinary, limit):
        self.ordinary = ordinary
        self.limit = limit
        self.receipt = None

    def __call__(
        self,
        model,
        tok,
        request,
        max_tokens,
        *,
        progress=None,
        stop_ids=(),
        tool_prefix=None,
    ):
        if tool_prefix is not None:
            raise ValueError(
                "The draft and action use independent request-owned native caches"
            )
        ids = tok.encode(request, add_special_tokens=False)
        if (
            not ids
            or not 1 <= max_tokens <= 4096
            or len(ids) + max_tokens > model.args.max_position_embeddings
        ):
            raise ValueError(
                "First-turn request exceeds the supported generation budget"
            )
        term = set(stop_ids) | set(tok.eos_token_ids)
        boundaries = term | {
            tok.convert_tokens_to_ids("</think>"),
            tok.convert_tokens_to_ids("<|im_start|>"),
        }
        mx.synchronize()
        at = perf_counter()
        start, boundary, first = draft(
            self.ordinary, ids, min(self.limit, max_tokens - 1), boundaries, progress
        )
        self.receipt = {"ids": start, "proposed_boundary": boundary}
        out = list(start)
        cache, hidden = prefill(model, ids + out, progress)
        for i in range(len(out), max_tokens):
            token = int(model.project(hidden).reshape(-1).argmax().item())
            if first is None:
                first = perf_counter()
            out.append(token)
            if progress:
                progress(len(ids) + len(out))
            if token in term or i + 1 == max_tokens:
                break
            hidden = model.hidden(mx.array([[token]]), cache)[:, -1:]
        return {
            "token_ids": out,
            "elapsed_s": perf_counter() - at,
            "ttft_s": first - at,
            "peak_bytes": mx.get_peak_memory(),
        }
