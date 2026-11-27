from loopqwen.research.shared_query_transfer import score


def test_report_scoring_requires_an_actual_nonidentity_calculation_and_consistent_final():
    w = {
        "family": "report",
        "gold": "7",
        "tolerance": "0",
        "observed": {"results": ["7"]},
    }

    def result(expression, value, answer):
        t = {
            "call": {"name": "calculate", "arguments": {"steps": [expression]}},
            "observation": {"value": value, "results": [value]},
        }
        return score(w, {"status": "answered", "turns": [t], "answer": answer})

    assert result("3+4", "7", "7")["success"]
    assert not result("7+0", "7", "7")["arithmetic_path"]
    assert not result("3+3", "6", "7")["success"]
    wrong = result("3+4", "7", "12")
    assert wrong["arithmetic_path"] and not wrong["success"]


def test_full_explanations_can_state_the_conclusion_and_cannot_gain_keyword_credit():
    w = {"family": "strategy", "label": False}

    def answered(text):
        return score(w, {"status": "answered", "answer": text, "turns": []})["success"]

    assert answered("**No**, the dates do not overlap.")
    assert not answered("Yes, even though an earlier account said no.")
    assert not answered("The notes contain yes and no.")
    assert not score(w, {"status": "token_limit", "answer": "No.", "turns": []})[
        "success"
    ]
