"""Pin the consecutive offline numeric prefix of the publisher's BFCL execution set."""

import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from jsonschema import validate

from loopqwen.evaluation.executable_tools import oracle, tools_for
from loopqwen.io import digest, save

if __name__ == "__main__":
    repo = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    rev = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"
    path = Path(
        hf_hub_download(
            repo, "BFCL_v3_exec_simple.json", revision=rev, repo_type="dataset"
        )
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()[:22]]
    for row in rows:
        call, value = oracle(row)
        tool = tools_for(row)[0]
        assert call["name"] == tool.name
        validate(call["arguments"], tool.parameters)
        assert abs(tool.callback(**call["arguments"])["value"] - value) <= max(
            abs(value) * 1e-6, 1e-10
        )
    save(
        "research/data/tool-exec-bfcl-01.json",
        dict(repository=repo, revision=rev, sha256=digest(path), rows=rows),
    )
