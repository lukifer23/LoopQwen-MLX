"""Use ordinary depth for early deliberation and recurrent depth for unfinished turns."""

import hashlib
from contextlib import ExitStack
from copy import deepcopy
from tempfile import TemporaryDirectory
from time import perf_counter

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.tuner.lora import LoRALinear

from . import REVISION
from .looped_model import LoopedModel
from .recurrence import Span


def ordinary_view(model, original_weights=None):
    arrays = {id(v): v for _, v in tree_flatten(model.text.parameters())}
    text = deepcopy(model.text, arrays)
    for layer in text.layers:
        for name, module in list(layer.named_modules()):
            if isinstance(module, LoRALinear):
                parent, key = name.rsplit(".", 1)
                setattr(getattr(layer, parent), key, module.linear)
    if original_weights is not None:
        parameters = dict(tree_flatten(text.parameters()))
        if not original_weights or any(
            k not in parameters
            or parameters[k].shape != v.shape
            or parameters[k].dtype != v.dtype
            for k, v in original_weights.items()
        ):
            raise ValueError("Original projection set differs from the merged decoder")
        text.load_weights(list(original_weights.items()), strict=False)
    ordinary = LoopedModel(text, Span(passes=1))
    ordinary.eval()
    return ordinary


def attach(model, policy, *, original_weights=None, directory=None):
    if (
        policy["revision"] != REVISION
        or policy["version"] not in (1, 2)
        or type(policy["switch_after"]) is not int
        or policy["switch_after"] < 1
        or policy.get("finisher_depth", 2) not in (1, 2)
        or model.span.passes != policy.get("finisher_depth", 2)
    ):
        raise ValueError("Unsupported recurrent deliberation policy")
    if policy["version"] == 2:
        path = directory / "exit-readout.safetensors"
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != policy["readout_sha256"]:
                raise ValueError("Learned exit weights differ")
        weights = mx.load(str(path))
        if (
            set(weights) != {"weight"}
            or weights["weight"].shape != (model.args.hidden_size + 2,)
            or not mx.all(mx.isfinite(weights["weight"])).item()
        ):
            raise ValueError("Invalid learned exit readout")
        object.__setattr__(model, "exit_readout", weights["weight"])
    ordinary = ordinary_view(model, original_weights)
    # The dense view shares immutable arrays and is deliberately outside the parameter tree.
    object.__setattr__(model, "ordinary_depth", ordinary)
    object.__setattr__(model, "deliberation_policy", dict(policy))


def turn_open(ids, opening, closing):
    for tok in reversed(ids):
        if tok in (opening, closing):
            return tok == opening
    return False


def greedy(
    model,
    tokenizer,
    ids,
    cap,
    *,
    before=None,
    progress=None,
    stop_ids=(),
    offload_dir=None,
    prefill_block=128,
):
    """Replay the whole causal prefix once if a thinking turn reaches the depth boundary."""
    model.eval()
    opening, closing = (
        tokenizer.encode(k, add_special_tokens=False)[0]
        for k in ("<think>", "</think>")
    )
    policy = getattr(model, "deliberation_policy", None)
    early = bool(policy and turn_open(ids, opening, closing))
    stage = model.ordinary_depth if early else model
    switch = policy["switch_after"] if early else None
    out = []
    switched = False
    closed = False
    readout = getattr(model, "exit_readout", None)
    scratch = 0
    first = None
    mx.synchronize()
    start = perf_counter()
    with ExitStack() as resources:
        cache = disk = None
        for i in range(cap):
            if before:
                before()
            if i == switch and not closed:
                stage = model
                switched = True
                cache = disk = None
                resources.close()
            if i == 0 or (switched and i == switch):
                history = ids + out
                if offload_dir is not None:
                    from .offloaded_prefill import OffloadedPrefill

                    folder = resources.enter_context(
                        TemporaryDirectory(dir=offload_dir)
                    )
                    disk = OffloadedPrefill(stage, folder)
                    hidden = disk.prefill_hidden(history, progress)
                    scratch = max(scratch, disk.scratch_bytes)
                else:
                    cache = stage.make_cache()
                    for j in range(0, len(history), prefill_block):
                        x = mx.array([history[j : j + prefill_block]])
                        h = stage.hidden(x, cache)
                        if j + prefill_block < len(history):
                            mx.eval(h)
                            if progress:
                                progress(j + x.shape[1])
                    hidden = h[:, -1:]
            else:
                x = mx.array([[out[-1]]])
                hidden = (
                    disk.decode_hidden(x) if disk else stage.hidden(x, cache)[:, -1:]
                )
            logits = stage.project(hidden)
            if switched and not closed and readout is not None:
                delta = (
                    hidden.astype(mx.float32) @ readout[:-2]
                    + readout[-2] * (i / switch)
                    + readout[-1]
                )
                logits = logits.astype(mx.float32).at[..., closing].add(delta)
            tok = logits.reshape(-1).argmax().item()
            if first is None:
                first = perf_counter()
            out.append(tok)
            closed = closed or tok == closing
            if progress:
                progress(len(ids) + i)
            if tok in tokenizer.eos_token_ids or tok in stop_ids:
                break
    end = perf_counter()
    result = {
        "token_ids": out,
        "ttft_s": first - start,
        "elapsed_s": end - start,
        "recurrent_from": switch if switched else None,
    }
    if offload_dir is not None:
        result["scratch_bytes"] = scratch
    return result
