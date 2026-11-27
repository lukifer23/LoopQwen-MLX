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
