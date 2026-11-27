from contextlib import nullcontext
from tempfile import TemporaryDirectory
from time import perf_counter

import mlx.core as mx


from ..output_projection import project_output


def resident_cache(model, route, length, max_tokens, block, *, reserve_full_kv=False):
    cache = model.make_cache()
    page_slots = {}
    if reserve_full_kv:
        from ..reserved_cache import reserve

        reserve(cache, length + max_tokens - 1)
    elif getattr(model, "packed_register_prefill", False):
        from ..page_prefill import register_cache

        page_slots = register_cache(model, route, length, max_tokens, block, cache)
    return cache, page_slots


def advance_cache(model, chunk, cache):
    """Materialize the physical histories of an intermediate resident prefill block."""
    populate = getattr(model, "populate_cache", model.hidden)
    populate(chunk, cache)
    mx.eval([slot.state for slot in cache.slots.values()])


def generate(
    model,
    tokenizer,
    prompt,
    max_tokens=24,
    *,
    progress=None,
    stop_ids=(),
    offload_dir=None,
    staged_context_dir=None,
    document_prefix=None,
    prefill_block=128,
    tool_prefix=None,
    reserve_full_kv=False,
    record_kv_history=False,
    tool_grammar=None,
    prefill_cache_state=False,
    return_prompt_ids=False,
):
    if type(prefill_block) is not int or prefill_block < 1:
        raise ValueError("Prefill block must be a positive integer")
    if document_prefix is not None and (
        offload_dir is not None
        or staged_context_dir is not None
        or tool_prefix is not None
        or reserve_full_kv
        or prefill_block != 128
        or any(
            getattr(model, k, None) is not None
            for k in ("context_policy", "deliberation_policy", "attention_reader")
        )
    ):
        raise ValueError(
            "Document reuse requires the original fixed model and a single cache owner"
        )
    if staged_context_dir is not None:
        if offload_dir is not None or any(
            getattr(model, key, None) is not None
            for key in ("context_policy", "deliberation_policy", "attention_reader")
        ):
            raise ValueError("Grouped full-history staging requires one fixed decoder")
        offload_dir = staged_context_dir
        tool_prefix = None
    if offload_dir is not None and prefill_block != 128:
        raise ValueError("Disk-backed prefill requires the validated 128-token block")
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if (
        not ids
        or not 1 <= max_tokens <= 4096
        or len(ids) + max_tokens > model.args.max_position_embeddings
    ):
        raise ValueError(
            "Supply a nonempty prompt within the context window and 1..4096 output tokens."
        )
    if reserve_full_kv and (offload_dir is not None or tool_prefix is not None):
        raise ValueError("Full-history reservation requires a fresh resident cache")
    if tool_prefix is not None and (
        offload_dir is not None
        or any(
            getattr(model, key, None) is not None
            for key in ("context_policy", "deliberation_policy", "attention_reader")
        )
    ):
        raise ValueError(
            "Tool prefix reuse requires a fixed model without prompt-dependent attention"
        )
    route = None
    routed_at = None
    policy = getattr(model, "context_policy", None)
    if policy is not None:
        mx.synchronize()
        routed_at = perf_counter()
        model, route = policy.choose(model, ids, max_tokens)
    if getattr(model, "deliberation_policy", None):
        if reserve_full_kv:
            raise ValueError("Deliberation uses separately scheduled caches")
        from ..deliberation import greedy

        val = greedy(
            model,
            tokenizer,
            ids,
            max_tokens,
            progress=progress,
            stop_ids=stop_ids,
            offload_dir=offload_dir,
            prefill_block=prefill_block,
        )
        if routed_at is not None:
            total = perf_counter() - routed_at
            val["ttft_s"] += total - val["elapsed_s"]
            val["elapsed_s"] = total
            val["context_route"] = route
        tokens = val["token_ids"]
        return dict(
            **val,
            prompt=prompt,
            text=tokenizer.decode(tokens),
            prompt_tokens=len(ids),
            **({"prompt_ids": ids} if return_prompt_ids else {}),
            greedy=True,
            seed=0,
            max_tokens=max_tokens,
            prefill_tps=len(ids) / val["ttft_s"],
            decode_tps=(len(tokens) - 1) / (val["elapsed_s"] - val["ttft_s"])
            if len(tokens) > 1
            else None,
            peak_bytes=mx.get_peak_memory(),
            active_bytes=mx.get_active_memory(),
        )
    workspace = (
        TemporaryDirectory(dir=offload_dir)
        if offload_dir is not None
        else nullcontext()
    )
    with workspace as directory:
        disk = None
        cache = None
        page_slots = {}
        if document_prefix is not None:
            disk = document_prefix.branch(model, ids)
        elif directory is None:
            cache, page_slots = resident_cache(
                model,
                route,
                len(ids),
                max_tokens,
                prefill_block,
                reserve_full_kv=reserve_full_kv,
            )
        else:
            from ..offloaded_prefill import OffloadedPrefill

            disk = OffloadedPrefill(
                model, directory, group_attention=staged_context_dir is not None
            )
        inputs = mx.array([ids])
        model.eval()
        mx.synchronize()
        start = perf_counter() if routed_at is None else routed_at
        section = None
        reader = getattr(model, "attention_reader", None)
        if reader is not None:
            section = reader.select(model, ids)
            if section is not None:
                from ..attention_memory import attach_bias

                attach_bias(cache if disk is None else disk.cache, section)
        # Bound prefill activation memory. Each chunk advances every site once.
        if progress is not None:
            progress(0)
        if tool_prefix is not None:
            offset = tool_prefix.begin(model, ids, prefill_block, cache)
            inputs = inputs[:, offset:]
        state_blocks = 0
        while disk is None and inputs.shape[1] > prefill_block:
            chunk = inputs[:, :prefill_block]
            if prefill_cache_state:
                advance_cache(model, chunk, cache)
                state_blocks += 1
            elif hasattr(model, "hidden"):
                mx.eval(model.hidden(chunk, cache))
            else:
                mx.eval(model(chunk, cache=cache)[:, -1])
            inputs = inputs[:, prefill_block:]
            if tool_prefix is not None:
                tool_prefix.capture(model, ids, len(ids) - inputs.shape[1], cache)
            if progress is not None:
                progress(len(ids) - inputs.shape[1])
        output = []
        for step in range(max_tokens):
            if disk is not None:
                if step:
                    logits = disk.decode(inputs)
                elif document_prefix is None:
                    logits = disk.prefill(ids, progress)
                else:
                    logits = document_prefix.complete_opening(disk, ids, progress)
            elif hasattr(model, "last_hidden"):
                logits = project_output(model, model.last_hidden(inputs, cache), cache)
            elif hasattr(model, "hidden"):
                logits = project_output(
                    model, model.hidden(inputs, cache)[:, -1:], cache
                )
            else:
                logits = model(inputs, cache=cache)
            if disk is not None and step == 0 and section is not None:
                # Prefill replaced full KV slots with disk-backed prefixes.
                attach_bias(disk.cache, section)
            token = mx.argmax(logits[:, -1], -1).item()
            if tool_grammar is not None:
                token = tool_grammar.choose(logits[0, -1], token, output)
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
    if document_prefix is not None:
        reused = len(document_prefix.ids)
        result.update(
            reused_prompt_tokens=reused,
            new_prompt_tokens=len(ids) - reused,
            prefill_tps=(len(ids) - reused) / result["ttft_s"],
        )
    if return_prompt_ids:
        result["prompt_ids"] = ids
    precision = getattr(model, "weight_precision_receipt", None)
    if precision is not None:
        inputs_used = model.text.model.embed_tokens
        result["weight_precision"] = {
            **precision,
            "actual_projections": model.projection_calls,
            "prompt_embedding_rows": inputs_used.initial_rows,
            "final_embedding_rows": inputs_used.rows.shape[0],
        }
    if tool_prefix is not None:
        result["reused_prompt_tokens"] = tool_prefix.reused
    if prefill_cache_state:
        result["cache_state_blocks"] = state_blocks
    if reserve_full_kv:
        from ..reserved_cache import ReservedKVCache

        result["reserved_sites"] = [
            dict(layer=i, turn=p, capacity=slot.keys.shape[2], offset=slot.offset)
            for (i, p), slot in cache.slots.items()
            if isinstance(slot, ReservedKVCache)
        ]
    if page_slots:
        result["prefill_policy"] = "direct-packed-register-v1"
        if record_kv_history:
            result["page_sites"] = [
                dict(layer=i, turn=p, **slot.report())
                for (i, p), slot in page_slots.items()
            ]
    if route is not None:
        result["context_route"] = route
    if section is not None:
        result["attention_memory"] = section
    if record_kv_history:
        from mlx_lm.models.cache import KVCache, QuantizedKVCache

        from ..offloaded_prefill import PrefixOnDisk

        history_cache = disk.cache if disk is not None else cache
        result["kv_history"] = [
            dict(
                layer=i,
                turn=p,
                offset=slot.offset,
                retained=slot.physical_length()
                if isinstance(slot, PrefixOnDisk)
                else slot.keys_and_values()[0][0].shape[2]
                if isinstance(slot, QuantizedKVCache)
                else slot.keys_and_values()[0].shape[2],
            )
            for (i, p), slot in history_cache.slots.items()
            if isinstance(slot, (KVCache, QuantizedKVCache))
        ]
    if disk is not None:
        result["scratch_bytes"] = disk.scratch_bytes
    return result
