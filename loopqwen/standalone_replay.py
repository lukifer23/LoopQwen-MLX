"""Serve a sequence of independent requests from one local native checkpoint."""

import argparse
import json
import os
from pathlib import Path
from time import monotonic

import mlx.core as mx

from .calculator_tool import tool
from .csv_tools import tools as csv_tools
from .io import read
from .tool_cli import open_assistant


def replay(checkpoint, requests, max_tokens):
    start = monotonic()
    assistant = open_assistant(
        argparse.Namespace(checkpoint=checkpoint, tool_format="qwen")
    )
    mx.eval(assistant._tool_model.parameters())
    print(
        json.dumps(
            {
                "pid": os.getpid(),
                "loaded_seconds": monotonic() - start,
                "loaded_bytes": mx.get_active_memory(),
            }
        ),
        flush=True,
    )
    for r in requests:
        providers = []
        if r.get("calculator"):
            providers.append(tool())
        if r.get("csv"):
            providers.extend(csv_tools([Path(r["csv"])], keyed=True))
        chat = assistant.conversation(providers)
        for question in r["prompts"]:
            tick = monotonic()
            mx.reset_peak_memory()
            result = chat.respond(
                question,
                max_tokens=max_tokens,
                max_turns=3,
                reuse_prefix=False,
                prefill_cache_state=True,
                record_kv_history=True,
            )
            print(
                json.dumps(
                    {
                        "elapsed": monotonic() - tick,
                        "peak": mx.get_peak_memory(),
                        "result": result,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    args = parser.parse_args()
    replay(args.checkpoint, read(args.requests), args.max_tokens)
