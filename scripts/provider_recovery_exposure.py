"""Count explicit provider failure histories and subsequent positive action targets."""

import json
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest
from scripts.provider_recovery_material import ROOT
from scripts.table_policy_material import persist


def count(path):
    turns = unpack(path)
    with_tool = with_error = next_action = 0
    for r in turns:
        observations = [m["content"] for m in r["messages"] if m["role"] == "tool"]
        with_tool += bool(observations)
        found = False
        for value in observations:
            try:
                obj = json.loads(value) if isinstance(value, str) else value
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "error" in obj:
                found = True
        with_error += found
        next_action += bool(found and r["target"].get("tool_calls"))
    return {
        "source": str(path),
        "sha256": digest(path),
        "turns": len(turns),
        "with_prior_tool_return": with_tool,
        "with_structured_error": with_error,
        "error_followed_by_supervised_action": next_action,
    }


def run():
    src = ROOT.parent
    paths = [
        src / "program-language" / "training.json.gz",
        src / "contrastive-worlds" / "training.json.gz",
        src / "shared-query" / "training.json.gz",
        *[
            ROOT / name
            for name in (
                "training.json.gz",
                "csv-training.json.gz",
                "csv-role-training.json.gz",
            )
        ],
    ]
    data = [count(p) for p in paths]
    persist(
        ROOT / "exposure.json",
        {
            "definition": "Count each target once if its actual input contains a prior tool-role message with a JSON object containing an error field. These repositories do not enumerate the public Qwen donor's pretraining, and general text may describe failure in other forms.",
            "cohorts": data,
        },
    )
    return data


if __name__ == "__main__":
    for r in run():
        print(r["source"], r["turns"], r["with_structured_error"])
