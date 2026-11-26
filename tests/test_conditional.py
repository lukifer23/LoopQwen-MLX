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


def test_final_mixed_panel_preserves_reserved_items_and_eight_distinct_articles():
    from loopqwen.io import read

    original = read("research/data/conditional-language-01.json")
    panel = read("research/data/conditional-language-02.json")
    assert len(panel) == len({r["id"] for r in panel}) == 80
    assert [r for r in panel if r.get("corpus") != "QuAC validation"] == original
    reading = [r for r in panel if r["family"] == "reading"]
    assert len({r["source"] for r in reading}) == len(reading) == 8
