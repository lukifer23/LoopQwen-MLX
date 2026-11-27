from loopqwen.research.cross_view_study import grade


def test_open_answer_must_use_real_return_and_distinguish_questions():
    call = {
        "name": "csv_lookup",
        "arguments": {
            "document": "notes.csv",
            "key_column": "title",
            "key": "Report",
            "value_column": "passage",
        },
    }
    obs = {"value": "Ava filed the report. Lin checked the figures."}
    field = {
        "family": "file",
        "calls": [call],
        "observation": obs,
        "gold": "Lin",
        "other": "Ava",
    }

    def answer(text, result=obs, status="answered"):
        return grade(
            field,
            0,
            {
                "status": status,
                "answer": text,
                "turns": [{"call": call, "observation": result}],
            },
        )

    assert answer("Lin.")["success"] and answer("Lin.")["exact"]
    assert (
        answer("The reviewer was Lin.")["success"]
        and not answer("The reviewer was Lin.")["exact"]
    )
    assert not answer("Ava or Lin.")["correct_answer"]
    assert not answer("Lindsey.")["correct_answer"]
    assert not answer("Lin.", {"value": "A different paragraph."})["success"]
    assert not answer("Lin.", status="token_limit")["completed"]
