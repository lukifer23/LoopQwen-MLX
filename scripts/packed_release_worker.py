"""Execute a crossed release audit from the exported runtime in a clean interpreter."""

import hashlib
import json
import resource
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx


def fingerprint(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def book(model, row, check):
    # Make the route decision before any evaluated continuation is available.
    ids = row["prompt_ids"]
    first_scored = len(ids) - 256
    at = perf_counter()
    view, route = model.context_policy.choose(model, ids[:first_scored], 256)
    from runtime.evaluation.generation_eval import resident_cache

    cache, pages = resident_cache(view, route, first_scored, 256, 128)
    if pages:
        raise ValueError("The register policy activated on natural prose")
    x = mx.array([ids[:-1]])
    losses, predicted = [], []
    for start in range(0, x.shape[1], 128):
        end = min(start + 128, x.shape[1])
        h = view.hidden(x[:, start:end], cache)
        mx.eval(h)
        # Each logit at position t is compared only to the actual token t+1.
        lo = max(start, first_scored - 1)
        for p in range(lo, end, 16):
            stop = min(end, p + 16)
            z = view.project(h[0, p - start : stop - start]).astype(mx.float32)
            gold = mx.array(ids[p + 1 : stop + 1])
            selected = mx.take_along_axis(z, gold[:, None], axis=-1)[:, 0]
            losses.extend((mx.logsumexp(z, axis=-1) - selected).tolist())
            predicted.extend(mx.argmax(z, axis=-1).tolist())
        check()
    from mlx_lm.models.cache import KVCache

    records = [
        [layer, turn, s.offset, s.keys_and_values()[0].shape[2]]
        for (layer, turn), s in cache.slots.items()
        if isinstance(s, KVCache)
    ]
    return {
        "id": row["id"],
        "input_sha256": fingerprint(ids),
        "tokens": len(ids),
        "route": route,
        "nll": losses,
        "predicted": predicted,
        "retained": records,
        "seconds": perf_counter() - at,
        "peak": mx.get_peak_memory(),
    }


def main():
    bundle, source, block, result, peak_limit = sys.argv[1:]
    sys.path.insert(0, bundle)
    from importlib.metadata import metadata, version

    from runtime.evaluation.generation_eval import generate
    from runtime.tool_model import Assistant
    from runtime.tool_session import Tool

    material = json.loads(Path(source).read_bytes())
    block = int(block)
    at = perf_counter()
    assistant = Assistant(Path(bundle))
    assistant._load_language()
    tok = assistant.tokenizer
    cold = perf_counter() - at

    def check(*_):
        if mx.get_peak_memory() > float(peak_limit) * 2**30:
            raise MemoryError("Isolated release MLX limit")

    def generation(row, cap):
        ids = row["prompt_ids"]
        text = tok.decode(ids)
        if tok.encode(text, add_special_tokens=False) != ids:
            raise ValueError("Release token stream changed")
        mx.reset_peak_memory()
        r = generate(
            assistant.language,
            tok,
            text,
            cap,
            stop_ids=[tok.convert_tokens_to_ids("<|im_end|>")],
            record_kv_history=True,
            progress=check,
        )
        return dict(
            id=row.get("id", "register"),
            input_sha256=fingerprint(ids),
            **{k: v for k, v in r.items() if k not in ("prompt", "text")},
        )

    rows = material["short"][block * 6 : (block + 1) * 6]
    first = [generation(r, r["cap"]) for r in rows[:3]]
    register = generation(material["register"]["panel"][block], 96)
    extra = {}
    if "medium_register" in material:
        extra["medium_register"] = generation(
            material["medium_register"]["panel"][block], 96
        )
    mx.reset_peak_memory()
    prose = book(assistant.language, material["books"][block], check)
    last = [generation(r, r["cap"]) for r in rows[3:]]
    package = ("mlx", "numpy")[block]
    callbacks = (
        lambda distribution: {"version": version(distribution)},
        lambda distribution: {"summary": metadata(distribution).get("Summary", "")},
    )
    tools = [
        Tool(f["name"], f["description"], f["parameters"], fn)
        for f, fn in zip(material["tools"], callbacks, strict=True)
    ]
    question = f"What version of the Python distribution {package} is installed in this environment? Use the available tool."
    trajectory = assistant.respond(question, tools, max_tokens=256, progress=check)
    if "unicode" in material:
        from runtime.tool_session import prompt

        labels = [w["text"] for w in material["unicode"]["rows"]]
        ids = assistant.tokenizer._tokenizer(labels, add_special_tokens=False)[
            "input_ids"
        ]
        q = material["unicode"]["question"]["question"]
        assistant._load_language()
        text = prompt(assistant.tokenizer, [{"role": "user", "content": q}])
        res = generate(
            assistant.language,
            assistant.tokenizer,
            text,
            384,
            stop_ids=[assistant.tokenizer.convert_tokens_to_ids("<|im_end|>")],
            progress=check,
        )
        raw = assistant.tokenizer.decode(res["token_ids"][:-1])
        done = res["token_ids"][-1] in (
            set(assistant.tokenizer.eos_token_ids)
            | {assistant.tokenizer.convert_tokens_to_ids("<|im_end|>")}
        )
        extra["unicode"] = {
            "source": fingerprint(labels),
            "label_ids": ids,
            "question": {
                "initial_ids": assistant.tokenizer.encode(
                    text, add_special_tokens=False
                ),
                "token_ids": res["token_ids"],
                "text": res["text"],
                "answer": raw.partition("</think>")[2].strip().lower()
                if done
                else None,
            },
        }
    if any(k == "loopqwen" or k.startswith("loopqwen.") for k in sys.modules):
        raise RuntimeError("Isolated process imported checkout inference code")
    Path(result).write_text(
        json.dumps(
            {
                "cold": cold,
                "stream_seconds": perf_counter() - at,
                "high_water_rss_bytes": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
                "short": first + last,
                "register": register,
                **extra,
                "book": prose,
                "tool": trajectory,
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
