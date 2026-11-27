from loopqwen.research.file_evidence import grade


def test_judgment_needs_the_actual_changed_cell():
    call = {
        "name": "csv_lookup",
        "arguments": {
            "document": "clippings.csv",
            "key_column": "article",
            "key": "Sample",
            "value_column": "excerpt",
        },
    }
    row = {
        "family": "file",
        "gold": "yes",
        "calls": [call],
        "evidence": "The new entry.",
    }

    def answer(returned):
        return {
            "status": "answered",
            "answer": "Yes.",
            "turns": [{"call": call, "observation": {"value": returned}}],
        }

    assert grade(row, 0, answer("The new entry."))["success"]
    stale = grade(row, 0, answer("The old entry."))
    assert stale["correct_answer"] and not stale["success"]
    missing = grade(row, 0, {"status": "answered", "answer": "Yes.", "turns": []})
    assert missing["correct_answer"] and not missing["success"]
    assert grade(
        dict(row, family="inline"),
        0,
        {"status": "answered", "answer": "Yes.", "turns": []},
    )["success"]
