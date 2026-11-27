"""Qualify the complete SQL scoring rule using original human replies and actual callback results."""

from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen.io import digest, read, save
from loopqwen.research.integrated_assistance import db_score, providers
from scripts.dialogue_database_material import PANEL


def audit():
    cases = []
    with TemporaryDirectory() as tmp:
        for w in read(PANEL)["rows"]:
            _, tool = providers(w, Path(tmp))
            args = {"database_name": w["source"] + ".sqlite", "statement": w["sql"]}
            event = {
                "call": {"name": tool.name, "arguments": args},
                "observation": tool.callback(**args),
            }
            answer = w["human_answer"]
            tr = {"status": "answered", "turns": [event, {}], "answer": answer}
            correct = db_score(w, tr)
            no_tool = db_score(w, {**tr, "turns": [{}]})
            assert (
                correct["success"]
                and not no_tool["success"]
                and no_tool["correct_answer"]
            )
            cases.append(
                {
                    "id": w["id"],
                    "original_reply": answer,
                    "original_expert": correct,
                    "visible_answer_alone": no_tool,
                }
            )
    report = {
        "cases": cases,
        "panel": digest(PANEL),
        "expert": sum(c["original_expert"]["success"] for c in cases),
        "no_tool": sum(c["visible_answer_alone"]["success"] for c in cases),
    }
    path = Path("docs/measurements/integrated-assistance-source-check.json")
    if path.is_file():
        assert read(path) == report
    else:
        save(path, report)
    print(
        report["expert"],
        "original analyst replies qualify;",
        report["no_tool"],
        "qualify without the database read",
    )


if __name__ == "__main__":
    audit()
