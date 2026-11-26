"""Run bounded multi-turn inference with caller-supplied JSON Schema tools."""

import json
import re
from dataclasses import dataclass
from time import perf_counter

from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry

from .evaluation.generation_eval import generate

SYSTEM = (
    'Use the available tools when needed. After thinking, to call a tool output exactly '
    '{"name":"function_name","arguments":{...}}. Call one tool at a time. '
    'Use its actual result. Otherwise answer the user directly and concisely. '
    'Ask for missing required information. Tool definitions follow.\n'
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
        return dict(name=self.name, description=self.description, parameters=self.parameters)


def initial_messages(question, tools):
    return [
        dict(role="system", content=SYSTEM + compact([t.definition() for t in tools])),
        dict(role="user", content=question),
    ]


def prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=True)


def object_pairs(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON key")
    return result


def decode_action(visible):
    if not visible.startswith(("{", "```")):
        return None
    obj = json.loads(visible, object_pairs_hook=object_pairs)
    if not isinstance(obj, dict) or set(obj) != {"name", "arguments"} or not isinstance(obj["name"], str) or not isinstance(obj["arguments"], dict):
        raise ValueError("Invalid tool call")
    return obj


def run(model, tokenizer, question, tools, *, max_tokens=128, max_turns=4, progress=None):
    """Return a visible answer and replayable trace. Only supplied callbacks may execute."""
    if not 1 <= max_turns <= 64:
        raise ValueError("max_turns must be between 1 and 64")
    catalog = {tool.name: tool for tool in tools}
    if len(catalog) != len(tools):
        raise ValueError("Tool names must be unique")
    validators = {}
    for tool in tools:
        if not callable(tool.callback) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", tool.name):
            raise ValueError("Invalid tool")
        Draft202012Validator.check_schema(tool.parameters)
        validators[tool.name] = Draft202012Validator(tool.parameters, registry=Registry())
    messages = initial_messages(question, tools)
    turns = []
    answer = None
    status = "turn_limit"
    start = perf_counter()
    for _ in range(max_turns):
        request = prompt(tokenizer, messages)
        out = generate(model, tokenizer, request, max_tokens=max_tokens, progress=progress)
        ids = out["token_ids"]
        row = dict(prompt_ids=tokenizer.encode(request, add_special_tokens=False), token_ids=ids,
                   seconds=out["elapsed_s"], ttft_s=out["ttft_s"], peak_bytes=out["peak_bytes"])
        turns.append(row)
        if ids[-1] not in tokenizer.eos_token_ids:
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
        messages.extend((dict(role="assistant", content=visible), dict(role="tool", content=payload)))
    return dict(answer=answer, status=status, turns=turns, seconds=perf_counter()-start)
