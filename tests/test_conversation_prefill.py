from types import SimpleNamespace

from loopqwen.tool_prefill import ConversationPrefill, ToolPrefix
from loopqwen.tool_session import Conversation, Tool


def test_follows_exact_tokens_and_graph_across_user_messages():
    graph = SimpleNamespace(_generation=object(), span=1, alpha_override=None)
    cache = SimpleNamespace(slots={0: SimpleNamespace(state=(0,))})
    history = ToolPrefix(max_tokens=None)
    local = ToolPrefix(max_tokens=4)
    combined = ConversationPrefill(history, local)
    ids = list(range(11))
    assert combined.begin(graph, ids, 2, cache) == 0
    for offset in range(2, 11, 2):
        cache.slots[0].state = (offset,)
        combined.capture(graph, ids, offset, cache)
    assert combined.begin(graph, ids + [44], 2, cache) == 10
    assert cache.slots[0].state == (10,)
    assert combined.begin(graph, ids[:4] + [90] * 9, 2, cache) == 4
    assert cache.slots[0].state == (4,)
    assert combined.begin(graph, [3] + ids, 2, cache) == 0
    graph._generation = object()
    assert combined.begin(graph, ids, 2, cache) == 0


def test_retained_states_are_owned_by_one_conversation():
    tool = Tool("clock", "Return a count", {"type": "object"}, dict)
    a = Conversation(object(), [tool], retain_prefix=True)
    b = Conversation(object(), [tool], retain_prefix=True)
    assert a._prefix is not b._prefix
    assert Conversation(object(), [tool])._prefix is None


def test_extended_prefix_advances_only_along_actual_shared_tokens():
    graph = SimpleNamespace(_generation=object(), span=1, alpha_override=None)
    cache = SimpleNamespace(slots={0: SimpleNamespace(state=(0,))})
    saved = ToolPrefix(max_tokens=8, extend=True)
    original = list(range(5))

    def consume(ids):
        offset = saved.begin(graph, ids, 2, cache)
        if offset:
            assert cache.slots[0].state == tuple(ids[:offset])
        for j in range(offset + 2, len(ids), 2):
            cache.slots[0].state = tuple(ids[:j])
            saved.capture(graph, ids, j, cache)
        return offset

    assert consume(original) == 0 and len(saved.tokens) == 4
    assert consume(list(range(7))) == 4 and len(saved.tokens) == 6
    assert consume([90] + list(range(11))) == 0 and saved.tokens == tuple(range(6))
    assert consume(list(range(13))) == 6 and len(saved.tokens) == 8
    assert consume(list(range(17))) == 8 and len(saved.tokens) == 8
    graph._generation = object()
    assert consume(list(range(17))) == 0
