"""Run bounded multi-turn inference with caller-supplied JSON Schema tools."""

import json
import re
from copy import deepcopy
from threading import Lock
from dataclasses import dataclass
from time import perf_counter

from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry

from .evaluation.generation_eval import generate

SYSTEM = (
    "Use the available tools when needed. After thinking, to call a tool output exactly "
    '{"name":"function_name","arguments":{...}}. Call one tool at a time. '
    "Use its actual result. Otherwise answer the user directly and concisely. "
    "Ask for missing required information. Tool definitions follow.\n"
)


def compact(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    callback: object

    def definition(self):
        return dict(
            name=self.name, description=self.description, parameters=self.parameters
        )


class Conversation:
    """Caller-owned conversation with a fixed tool catalog and ordered observations."""

    def __init__(self, assistant, tools, *, retain_prefix=False):
        from .tool_prefill import ToolPrefix

        self._assistant = assistant
        self._tools = tuple(
            Tool(t.name, t.description, deepcopy(t.parameters), t.callback)
            for t in tools
        )
        self._prefix = (
            ToolPrefix(max_tokens=None) if retain_prefix and self._tools else None
        )
        self._messages = (
            [system_message(self._tools)]
            if self._tools and getattr(assistant, "tool_format", "compact") == "compact"
            else []
        )
        self._active = Lock()

    @property
    def messages(self):
        return tuple(deepcopy(m) for m in self._messages)

    def respond(
        self,
        question,
        *,
        max_tokens=160,
        max_turns=4,
        progress=None,
        reuse_prefix=None,
        record_kv_history=False,
        prefill_cache_state=False,
    ):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("The next user message must contain text")
        if not self._active.acquire(blocking=False):
            raise RuntimeError("A response to this conversation is already in progress")
        try:
            return self._assistant.respond(
                question,
                self._tools,
                max_tokens=max_tokens,
                max_turns=max_turns,
                progress=progress,
                reuse_prefix=reuse_prefix,
                _messages=self._messages,
                **(
                    {"_conversation_prefix": self._prefix}
                    if self._prefix is not None
                    else {}
                ),
                **({"record_kv_history": True} if record_kv_history else {}),
                **({"prefill_cache_state": True} if prefill_cache_state else {}),
            )
        finally:
            self._active.release()


def system_message(tools):
    return {
        "role": "system",
        "content": SYSTEM + compact([t.definition() for t in tools]),
    }


def initial_messages(question, tools):
    return [
        system_message(tools),
        dict(role="user", content=question),
    ]


def prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )


def object_pairs(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON key")
    return result


def reject_constant(value):
    raise ValueError(f"Invalid JSON number {value}")


def decode_action(visible):
    if not visible.startswith(("{", "```")):
        return None
    obj = json.loads(
        visible, object_pairs_hook=object_pairs, parse_constant=reject_constant
    )
    if (
        not isinstance(obj, dict)
        or set(obj) != {"name", "arguments"}
        or not isinstance(obj["name"], str)
        or not isinstance(obj["arguments"], dict)
    ):
        raise ValueError("Invalid tool call")
    return obj


def run(
    model,
    tokenizer,
    question,
    tools,
    *,
    max_tokens=128,
    max_turns=4,
    progress=None,
    stop_at_turn_end=True,
    reuse_prefix=False,
    schema_guided=False,
    first_turn_decoder=None,
    observation_model=None,
    recover_callback=None,
    tool_format="compact",
    record_kv_history=False,
    prefill_cache_state=False,
    _messages=None,
    _conversation_prefix=None,
):
    """Return a visible answer and replayable trace. Only supplied callbacks may execute."""
    if tool_format not in ("compact", "qwen"):
        raise ValueError("Unknown tool format")
    native = tool_format == "qwen"
    if native:
        from . import qwen_tools

        if schema_guided:
            raise ValueError("The JSON grammar requires compact tool syntax")
        definitions = qwen_tools.declarations(tools)
        declared = {d["function"]["name"]: d["function"] for d in definitions}
    if not 1 <= max_turns <= 64:
        raise ValueError("max_turns must be between 1 and 64")
    catalog = {tool.name: tool for tool in tools}
    if len(catalog) != len(tools):
        raise ValueError("Tool names must be unique")
    validators = {}
    for tool in tools:
        if not callable(tool.callback) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_.-]{0,127}"
            if native
            else r"[A-Za-z_][A-Za-z0-9_.]{0,127}",
            tool.name,
        ):
            raise ValueError("Invalid tool")
        Draft202012Validator.check_schema(tool.parameters)
        validators[tool.name] = Draft202012Validator(
            tool.parameters, registry=Registry()
        )
    stop_ids = set(tokenizer.eos_token_ids)
    if stop_at_turn_end:
        stop_ids.add(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    if _messages is None:
        messages = (
            [{"role": "user", "content": question}]
            if native
            else initial_messages(question, tools)
        )
    else:
        messages = _messages
        messages.append(dict(role="user", content=question))
    from .tool_prefill import ConversationPrefill, ToolPrefix

    tool_prefix = ToolPrefix() if reuse_prefix else None
    if _conversation_prefix is not None:
        tool_prefix = ConversationPrefill(_conversation_prefix, tool_prefix)
    turns = []
    answer = None
    status = "turn_limit"
    start = perf_counter()
    grammar = None
    if schema_guided:
        from .tool_grammar import ToolGrammar

        grammar = ToolGrammar(tokenizer, model.args.vocab_size, tools, stop_ids)
    recovery_model = None
    for turn_number in range(max_turns):
        if grammar is not None:
            grammar.begin_turn()
        request = (
            qwen_tools.render(tokenizer, messages, definitions)
            if native
            else prompt(tokenizer, messages)
        )
        decoder = (
            first_turn_decoder
            if not turns and first_turn_decoder is not None
            else generate
        )
        active_model = (
            observation_model if turns and observation_model is not None else model
        )
        if recovery_model is not None:
            active_model = recovery_model
        prefix = tool_prefix
        if not turns and observation_model is not None:
            prefix = _conversation_prefix
        out = decoder(
            active_model,
            tokenizer,
            request,
            max_tokens=max_tokens,
            progress=progress,
            stop_ids=stop_ids,
            tool_prefix=prefix,
            **({"return_prompt_ids": True} if decoder is generate else {}),
            **({"record_kv_history": True} if record_kv_history else {}),
            **({"prefill_cache_state": True} if prefill_cache_state else {}),
            **({"tool_grammar": grammar} if grammar is not None else {}),
        )
        ids = out["token_ids"]
        # Use the input actually consumed by the native decoder. Custom first-turn
        # decoders can omit it and retain the original reconstruction behavior.
        prompt_ids = out.get("prompt_ids") if decoder is generate else None
        if prompt_ids is None:
            prompt_ids = tokenizer.encode(request, add_special_tokens=False)
        row = dict(
            prompt_ids=prompt_ids,
            token_ids=ids,
            seconds=out["elapsed_s"],
            ttft_s=out["ttft_s"],
            peak_bytes=out["peak_bytes"],
        )
        if recover_callback is not None:
            row["recovery_active"] = recovery_model is not None
        if record_kv_history:
            row.update(
                {
                    k: out[k]
                    for k in ("kv_history", "recurrent_from", "context_route")
                    if k in out
                }
            )
        if prefill_cache_state:
            row["cache_state_blocks"] = out["cache_state_blocks"]
        if tool_prefix is not None:
            row["reused_prompt_tokens"] = out.get("reused_prompt_tokens", 0)
        if grammar is not None:
            row["replaced_by_schema"] = grammar.replaced
        turns.append(row)
        if ids[-1] not in stop_ids:
            status = "token_limit"
            break
        raw = tokenizer.decode(ids[:-1])
        if "</think>" not in raw:
            status = "unfinished_thinking"
            break
        visible = raw.split("</think>", 1)[1].strip()
        row["visible"] = visible
        try:
            if native:
                actions = qwen_tools.parse(visible, declared)
            else:
                action = decode_action(visible)
                actions = [] if action is None else [action]
            if not actions:
                answer = visible
                status = "answered" if visible else "empty_answer"
                if answer and _messages is not None:
                    messages.append(dict(role="assistant", content=answer))
                break
            if len(actions) == 1:
                row["call"] = actions[0]
            else:
                row["calls"] = actions
            for action in actions:
                name = action["name"]
                if name not in catalog:
                    raise ValueError("Unknown tool")
                validators[name].validate(action["arguments"])
        except (ValueError, ValidationError):
            status = "invalid_call"
            if _messages is not None:
                messages.append(dict(role="assistant", content=visible))
            break
        assistant_turn = (
            qwen_tools.assistant_message(
                actions,
                visible.partition("<tool_call>")[0].strip(),
                raw.partition("</think>")[0].strip(),
            )
            if native
            else dict(role="assistant", content=visible)
        )
        observations = []
        tool_messages = []
        for action in actions:
            tool = catalog[action["name"]]
            try:
                payload = compact(tool.callback(**action["arguments"]))
            except Exception as exc:
                payload = compact({"error": type(exc).__name__})
                if (
                    recover_callback is not None
                    and recovery_model is None
                    and turn_number + 1 < max_turns
                ):
                    recovery_model = recover_callback(model, tool)
                    if recovery_model is not None and tool_prefix is not None:
                        tool_prefix = ToolPrefix()
            observations.append(json.loads(payload))
            tool_messages.append(dict(role="tool", content=payload))
        messages.extend([assistant_turn, *tool_messages])
        if len(observations) == 1:
            row["observation"] = observations[0]
        else:
            row["observations"] = observations
    return dict(
        answer=answer, status=status, turns=turns, seconds=perf_counter() - start
    )
