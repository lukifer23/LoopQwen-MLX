"""Check the counterfactual scoring boundary independently of model generation."""

from loopqwen.evaluation import venue_dialogues as venue
from loopqwen.io import read


def test_same_human_request_requires_distinct_live_answers_and_correct_query():
    material = read("research/data/tool-venue-dialogue-test.json")
    items, tools, judge = venue.workload(material)
    trace_rows = [r for r in items if r.get("service") == "attraction"]
    first = trace_rows[0]
    second = next(
        r
        for r in trace_rows
        if r["dialogue"] == first["dialogue"] and r["id"] != first["id"]
    )
    assert (
        first["user_request"] == second["user_request"]
        and first["gold"] != second["gold"]
    )
    args = venue.arguments(first["state"])

    def execute(row, args):
        tool = tools(row)[0]
        observed = tool.callback(**args)
        return {
            "answer": str(observed["values"][0]),
            "status": "answered",
            "turns": [
                dict(call={"name": tool.name, "arguments": args}, observation=observed)
            ],
        }

    for row in (first, second):
        trace = execute(row, args)
        assert judge(row, trace)["success"]
        wrong = execute(row, dict(args, filters=args["filters"][:-1]))
        wrong["answer"] = str(row["gold"])
        assert judge(row, wrong)["correct_answer"] and not judge(row, wrong)["success"]
    assert not judge(second, execute(first, args))["success"]
