"""Qwen's published function-call wire format and typed caller-owned messages."""

import json
import re

from .tool_session import compact, object_pairs, reject_constant

# Argument names can contain spaces, punctuation or Unicode. The publisher's tag
# syntax reserves angle brackets and line breaks in names.
OPEN = re.compile(r"<function=([^<>\r\n]+)>")
PARAM = re.compile(r"<parameter=([^<>\r\n]+)>\n?(.*?)\n?</parameter>", re.S)


def declarations(tools):
    return [{"type": "function", "function": t.definition()} for t in tools]


def render(tokenizer, messages, definitions):
    return tokenizer.apply_chat_template(
        messages,
        tools=definitions,
        tokenize=False,
        enable_thinking=True,
        add_generation_prompt=True,
    )


def typed(raw, schema):
    kind = schema.get("type", ())
    kinds = {kind} if isinstance(kind, str) else set(kind)
    if kinds == {"string"}:
        return raw
    try:
        value = json.loads(
            raw, object_pairs_hook=object_pairs, parse_constant=reject_constant
        )
    except (ValueError, RecursionError):
        if not kinds or "string" in kinds:
            return raw
        raise ValueError("Invalid typed function argument") from None
    compact(value)  # Reject numerical overflow even inside arrays and objects.
    return value


def parse(visible, available):
    """Accept one or more complete blocks. Validate every call before any callback."""
    before, marker, after = visible.partition("<tool_call>")
    if not marker:
        if re.search(r"</?(?:tool_call|function|parameter)(?:[=>\s]|$)", visible):
            raise ValueError("Malformed Qwen function block")
        return []
    if re.search(r"</?(?:tool_call|function|parameter)(?:[=>\s]|$)", before):
        raise ValueError("Malformed prefix before function")
    calls = []
    rest = marker + after
    while rest:
        if not rest.startswith("<tool_call>"):
            raise ValueError("Unexpected content after function call")
        rest = rest.removeprefix("<tool_call>").lstrip()
        match = OPEN.match(rest)
        if not match or match[1] not in available:
            raise ValueError("Unknown or invalid function")
        name = match[1]
        raw = rest[match.end() :].lstrip()
        schema = available[name].get("parameters", {})
        args = {}
        while raw.startswith("<parameter="):
            match = PARAM.match(raw)
            if not match or match[1] in args:
                raise ValueError("Invalid or duplicate function argument")
            props = schema.get("properties", {})
            extra = schema.get("additionalProperties", {})
            shape = props.get(match[1], extra if isinstance(extra, dict) else {})
            args[match[1]] = typed(match[2], shape)
            raw = raw[match.end() :].lstrip()
        if not raw.startswith("</function>"):
            raise ValueError("Unclosed function")
        raw = raw.removeprefix("</function>").lstrip()
        if not raw.startswith("</tool_call>"):
            raise ValueError("Unclosed tool call")
        rest = raw.removeprefix("</tool_call>").strip()
        calls.append({"name": name, "arguments": args})
    return calls


def assistant_message(calls, content="", thinking=""):
    return {
        "role": "assistant",
        "content": content,
        **({"reasoning_content": thinking} if thinking else {}),
        "tool_calls": [{"type": "function", "function": a} for a in calls],
    }
