import pytest
from mlx_lm.utils import load_tokenizer

from loopqwen import qwen_tools, tool_session
from loopqwen.model_loader import local_snapshot
from loopqwen.tool_session import Tool


def block(name, **kwargs):
    args = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in kwargs.items())
    return f"<tool_call>\n<function={name}>\n{args}</function>\n</tool_call>"


def test_qwen_parallel_calls_rebuild_published_template(monkeypatch):
    tok = load_tokenizer(local_snapshot())
    observed = []
    props = {
        "key": {"type": "string"},
        "value": {"type": "integer"},
        "items": {"type": "array"},
    }
    schema = {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }

    def record(**args):
        observed.append(args)
        return {"count": len(observed)}

    tool = Tool("record", "Record a reading", schema, record)
    a = block("record", key="null", value="9007199254740993", items='[1,{"ok":true}]')
    b = block("record", key="a\nb", value="7", items="[]")
    history = []
    requests = []

    def generate(_model, tok, text, **kw):
        visible = "I'll check both.\n" + a + "\n" + b if not requests else "2"
        requests.append(text)
        return dict(
            token_ids=tok.encode(
                "Consider both.\n</think>\n\n" + visible + "<|im_end|>",
                add_special_tokens=False,
            ),
            elapsed_s=1.0,
            ttft_s=0.1,
            peak_bytes=1,
        )

    monkeypatch.setattr(tool_session, "generate", generate)
    tr = tool_session.run(
        None, tok, "Read both", [tool], tool_format="qwen", _messages=history
    )
    assert observed == [
        dict(key="null", value=9007199254740993, items=[1, {"ok": True}]),
        dict(key="a\nb", value=7, items=[]),
    ]
    assert tr["answer"] == "2" and tr["turns"][0]["observations"] == [
        {"count": 1},
        {"count": 2},
    ]
    assert (
        history[1]["content"] == "I'll check both."
        and history[1]["reasoning_content"] == "Consider both."
    )
    assert requests[1] == qwen_tools.render(
        tok, history[:-1], qwen_tools.declarations([tool])
    )
    assert requests[1].count("<tool_response>") == 2


@pytest.mark.parametrize(
    "malformed",
    [
        block("missing", k="x"),
        block("read", k="a") + "suffix",
        "<tool_call><function=read><parameter=k>x</parameter><parameter=k>y</parameter></function></tool_call>",
        "<tool_call><function=read><parameter=k>x</parameter>",
    ],
)
def test_invalid_second_qwen_call_executes_neither(monkeypatch, malformed):
    tok = load_tokenizer(local_snapshot())
    values = []
    tool = Tool(
        "read",
        "Read",
        {"type": "object", "properties": {"k": {"type": "string"}}, "required": ["k"]},
        lambda k: values.append(k),
    )
    text = "</think>\n" + block("read", k="first") + "\n" + malformed + "<|im_end|>"
    monkeypatch.setattr(
        tool_session,
        "generate",
        lambda *a, **kw: dict(
            token_ids=tok.encode(text, add_special_tokens=False),
            elapsed_s=1,
            ttft_s=0.1,
            peak_bytes=1,
        ),
    )
    tr = tool_session.run(None, tok, "Read", [tool], tool_format="qwen")
    assert tr["status"] == "invalid_call" and values == []


def test_nested_nonfinite_numeric_data():
    with pytest.raises(ValueError):
        qwen_tools.typed("[1e9999]", {"type": "array"})
