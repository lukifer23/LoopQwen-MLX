"""Score chained package lookups using installed wheel metadata and observed versions."""

import re

from packaging.utils import canonicalize_name

from ..package_tools import tools


def workload(material, replay):
    graph = material["environment"]
    if not replay:
        from ..package_tools import snapshot

        if snapshot() != graph:
            raise ValueError(
                "Installed distribution metadata differs from the preregistered field environment"
            )
    return material["rows"], lambda _: tools(graph if replay else None), score


def score(row, trajectory):
    turns = [t for t in trajectory["turns"] if "observation" in t]
    calls = [
        (t["call"]["name"], canonicalize_name(t["call"]["arguments"]["distribution"]))
        for t in turns
    ]
    path = (
        calls
        == [("package_dependencies", row["root"]), ("package_version", row["target"])]
        and turns[0]["observation"] == {"dependencies": row["dependencies"]}
        and turns[1]["observation"] == {"version": row["version"]}
    )
    correct = bool(
        re.search(
            r"(?<![\w.])" + re.escape(row["version"]) + r"(?!\w|\.\d)",
            trajectory["answer"] or "",
        )
    )
    complete = trajectory["status"] == "answered"
    return dict(
        success=path and correct and complete,
        path=path,
        correct_answer=correct,
        completed=complete,
    )
