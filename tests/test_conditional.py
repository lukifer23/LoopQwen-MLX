from loopqwen.conditional import assistant_thinking


def test_routing_uses_the_current_assistant_turn():
    # These are the control tokens of the pinned Qwen tokenizer.
    assistant = [248045, 74455, 198, 248068, 198]
    user = [248045, 846, 198]
    assert assistant_thinking(user + [91, 248046] + assistant + [81, 90])
    assert not assistant_thinking([123, 248068, 198])
    assert not assistant_thinking(user + [42, 248068, 198])
    assert not assistant_thinking(assistant + [248069])
    assert not assistant_thinking(assistant + [248046] + user + [248068, 198])
    assert not assistant_thinking([248045, 74455, 198, 248069])
