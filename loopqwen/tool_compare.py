"""Compare read-only requests from the installed CLI on two actual local decoders."""

import gc
from time import perf_counter

import mlx.core as mx

from . import REVISION
from .ordinary_model import ordinary as base
from .identity import fingerprint
from .tool_cli import open_assistant, present
from .tool_model import Assistant
from .worker import acquire


class DenseAssistant(Assistant):
    """Run the same conversation protocol with the untouched ordinary-depth checkpoint."""

    def __init__(self, reuse, tool_format="compact", directory=None):
        self.tool_format = tool_format
        self.language, self.tokenizer, _ = base(directory)
        self._tool_model = self.language
        self.reuse_prefix = reuse
        self.dedicated = False


def compare(args, tools, options):
    acquire()
    prompts = [args.prompt, *args.follow_up]
    outputs = {}
    reuse = options["reuse_prefix"]
    base_source = None
    native_plain = False
    for side in ("recurrent", "base"):
        assistant = chat = None
        rows = []
        gc.collect()
        mx.clear_cache()
        for text in prompts:
            mx.synchronize()
            mx.reset_peak_memory()
            start = perf_counter()
            if assistant is None:
                if side == "recurrent":
                    assistant = open_assistant(args)
                    native_plain = getattr(assistant, "native_plain", False)
                    if reuse is None:
                        reuse = assistant.reuse_prefix
                else:
                    assistant = DenseAssistant(
                        reuse, getattr(args, "tool_format", "compact"), args.checkpoint
                    )
                    assistant.native_plain = native_plain
                    base_source = assistant.language.base_source
                chat = (
                    assistant.conversation(tools, retain_prefix=True)
                    if getattr(args, "retain_conversation_prefix", False)
                    else assistant.conversation(tools)
                )
            observed_at = {}

            def observe(consumed, times=observed_at, at=start):
                if consumed not in times:
                    times[consumed] = perf_counter() - at

            response = chat.respond(text, **options, progress=observe)
            mx.synchronize()
            elapsed = perf_counter() - start
            turns = response["turns"]
            initial = turns[0]["prompt_ids"]
            rows.append(
                {
                    "question": text,
                    "initial_tokens": len(initial),
                    "initial_sha256": fingerprint(initial),
                    "output_tokens": sum(len(t["token_ids"]) for t in turns),
                    "request_seconds": elapsed,
                    "first_token_seconds": observed_at[len(initial)],
                    "peak_mlx_bytes": mx.get_peak_memory(),
                    **present(response, args.trace),
                }
            )
        outputs[side] = rows
        del chat, assistant
    if (
        outputs["recurrent"][0]["initial_sha256"]
        != outputs["base"][0]["initial_sha256"]
    ):
        raise ValueError("Initial prompt tokens differ across assistants")
    return {
        "base_revision": REVISION,
        "base_source": base_source,
        **(
            {"tool_format": "qwen"}
            if getattr(args, "tool_format", "compact") == "qwen"
            else {}
        ),
        "recurrent_checkpoint": str(args.checkpoint),
        "tool_adapter": str(args.tool_adapter) if args.tool_adapter else None,
        **(
            {"session_policy": str(args.session_policy)}
            if getattr(args, "session_policy", None)
            else {}
        ),
        "reuse_prefix": reuse,
        **({"extend_tool_prefix": True} if options.get("extend_tool_prefix") else {}),
        **({"prefill_cache_state": True} if options.get("prefill_cache_state") else {}),
        "max_tokens_per_turn": options["max_tokens"],
        "max_turns": options["max_turns"],
        "order": ["recurrent", "base"],
        "initial_prompt_identical": True,
        "conversation_prompt_matches": [
            a["initial_sha256"] == b["initial_sha256"]
            for a, b in zip(outputs["recurrent"], outputs["base"], strict=True)
        ],
        "runs": outputs,
    }
