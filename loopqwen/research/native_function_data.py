"""Build official Qwen message targets and replay the enrolled read-only resources."""

import json

from .. import file_tools, package_tools, qwen_tools
from ..calculator_tool import tool
from ..evaluation import executable_tools
from ..tool_session import SYSTEM, compact


def catalog(row, panel=None):
    family = row["family"]
    if family == "public":
        return executable_tools.tools_for(row)
    if family == "packages":
        return package_tools.tools(panel["environment"] if panel else None)
    if family == "files":
        if panel is None:
            return file_tools.tools(row["paths"])
        return file_tools.definitions(
            lambda path, field: {field: panel["files"][path][field]}, row["paths"]
        )
    if family in ("ASDiv", "SVAMP"):
        return [tool()]
    raise ValueError(f"Unknown family {family}")


def from_compact(row):
    opening, *body = row["messages"]
    assert opening["role"] == "system" and opening["content"].startswith(SYSTEM)
    definitions = [
        {"type": "function", "function": s}
        for s in json.loads(opening["content"][len(SYSTEM) :])
    ]

    def message(text):
        try:
            v = json.loads(text)
        except ValueError:
            v = None
        if isinstance(v, dict) and set(v) == {"name", "arguments"}:
            return qwen_tools.assistant_message([v])
        return {"role": "assistant", "content": text}

    history = [message(m["content"]) if m["role"] == "assistant" else m for m in body]
    target = message(row["visible"])
    if row.get("thinking"):
        target["reasoning_content"] = row["thinking"]
    return dict(id=row["id"], tools=definitions, messages=history, target=target)


def token_pair(tok, row):
    prefix = qwen_tools.render(tok, row["messages"], row["tools"])
    full = tok.apply_chat_template(
        [*row["messages"], row["target"]],
        tools=row["tools"],
        tokenize=False,
        enable_thinking=True,
        add_generation_prompt=False,
    )
    if not full.startswith(prefix) or not full.endswith("<|im_end|>\n"):
        raise ValueError(
            "Reference assistant turn does not extend published generation prompt"
        )
    ids = tok.encode(prefix, add_special_tokens=False)
    # The generation prompt ends in a newline. A tokenizer may merge it with
    # another newline at the beginning of the answer when encoding a whole turn.
    # At inference this boundary is frozen. Encode only the generated suffix.
    reply = tok.encode(full[len(prefix) : -1], add_special_tokens=False)
    assert (
        reply[-1] == tok.convert_tokens_to_ids("<|im_end|>")
        and tok.decode(ids + reply) == full[:-1]
    )
    return ids, reply


def service_targets(row):
    tools = catalog(row)
    available = {t.name: t for t in tools}
    messages = []
    for i, (question, call) in enumerate(
        zip(row["questions"], row["calls"], strict=True)
    ):
        messages.append(dict(role="user", content=question))
        action = qwen_tools.assistant_message([call])
        yield dict(
            id=f"{row['id']}-{i}-call",
            tools=qwen_tools.declarations(tools),
            messages=list(messages),
            target=action,
        )
        observation = available[call["name"]].callback(**call["arguments"])
        field = next(iter(observation))
        result = str(observation[field])
        assert result == row["answers"][i]
        messages.extend([action, dict(role="tool", content=compact(observation))])
        answer = dict(role="assistant", content=result)
        yield dict(
            id=f"{row['id']}-{i}-answer",
            tools=qwen_tools.declarations(tools),
            messages=list(messages),
            target=answer,
        )
        messages.append(answer)
