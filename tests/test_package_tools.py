from loopqwen.evaluation.package_questions import score
from loopqwen.io import read
from loopqwen.package_tools import tools


def test_field_metadata_replay_needs_the_ordered_observations():
    material = read("research/data/tool-package-dependencies-01.json")
    dependencies, version = tools(material["environment"])
    for row in material["rows"]:
        first = {"name": dependencies.name, "arguments": {"distribution": row["root"]}}
        second = {"name": version.name, "arguments": {"distribution": row["target"]}}
        a = {"call": first, "observation": dependencies.callback(**first["arguments"])}
        b = {"call": second, "observation": version.callback(**second["arguments"])}
        tr = {"turns": [a, b], "answer": row["version"], "status": "answered"}
        assert score(row, tr)["success"]
        assert not score(row, dict(tr, turns=[b, a]))["success"]
        assert not score(row, dict(tr, answer="Unknown"))["success"]
