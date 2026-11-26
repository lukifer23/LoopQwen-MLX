"""Run bounded multi-turn inference with caller-supplied JSON Schema tools."""

import json
import re
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


def initial_messages(question, tools):
    return [
        dict(role="system", content=SYSTEM + compact([t.definition() for t in tools])),
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
):
    """Return a visible answer and replayable trace. Only supplied callbacks may execute."""
    if not 1 <= max_turns <= 64:
        raise ValueError("max_turns must be between 1 and 64")
    catalog = {tool.name: tool for tool in tools}
    if len(catalog) != len(tools):
        raise ValueError("Tool names must be unique")
    validators = {}
    for tool in tools:
        if not callable(tool.callback) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_.]{0,127}", tool.name
        ):
            raise ValueError("Invalid tool")
        Draft202012Validator.check_schema(tool.parameters)
        validators[tool.name] = Draft202012Validator(
            tool.parameters, registry=Registry()
        )
    stop_ids = set(tokenizer.eos_token_ids)
    if stop_at_turn_end:
        stop_ids.add(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    messages = initial_messages(question, tools)
    from .tool_prefill import ToolPrefix

    tool_prefix = ToolPrefix() if reuse_prefix else None
    turns = []
    answer = None
    status = "turn_limit"
    start = perf_counter()
    grammar = None
    if schema_guided:
        from .tool_grammar import ToolGrammar

        grammar = ToolGrammar(tokenizer, model.args.vocab_size, tools, stop_ids)
    for _ in range(max_turns):
        if grammar is not None:
            grammar.begin_turn()
        request = prompt(tokenizer, messages)
        decoder = (
            first_turn_decoder
            if not turns and first_turn_decoder is not None
            else generate
        )
        out = decoder(
            model,
            tokenizer,
            request,
            max_tokens=max_tokens,
            progress=progress,
            stop_ids=stop_ids,
            tool_prefix=tool_prefix,
            **({"tool_grammar": grammar} if grammar is not None else {}),
        )
        ids = out["token_ids"]
        row = dict(
            prompt_ids=tokenizer.encode(request, add_special_tokens=False),
            token_ids=ids,
            seconds=out["elapsed_s"],
            ttft_s=out["ttft_s"],
            peak_bytes=out["peak_bytes"],
        )
        if tool_prefix is not None:
            row["reused_prompt_tokens"] = out["reused_prompt_tokens"]
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
            action = decode_action(visible)
            if action is None:
                answer = visible
                status = "answered" if visible else "empty_answer"
                break
            row["call"] = action
            name = action["name"]
            if name not in catalog:
                raise ValueError("Unknown tool")
            validators[name].validate(action["arguments"])
        except (ValueError, ValidationError):
            status = "invalid_call"
            break
        try:
            result = catalog[name].callback(**action["arguments"])
            payload = compact(result)
        except Exception as exc:
            # Callback failures are returned to the model as observations.
            payload = compact({"error": type(exc).__name__})
        row["observation"] = json.loads(payload)
        messages.extend(
            (
                dict(role="assistant", content=visible),
                dict(role="tool", content=payload),
            )
        )
    return dict(
        answer=answer, status=status, turns=turns, seconds=perf_counter() - start
    )
