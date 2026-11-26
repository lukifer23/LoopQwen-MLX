"""Freeze the consecutive opening math questions from the publisher's BFCL v3 simple set."""

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from loopqwen.io import digest, save

REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
REV = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"


def material():
    files = [
        Path(hf_hub_download(REPO, name, revision=REV, repo_type="dataset"))
        for name in ("BFCL_v3_simple.json", "possible_answer/BFCL_v3_simple.json")
    ]
    requests, answers = [
        [json.loads(line) for line in f.read_text().splitlines()] for f in files
    ]
    answers = {r["id"]: r["ground_truth"] for r in answers}
    return dict(
        repository=REPO,
        revision=REV,
        sha256=[digest(f) for f in files],
        rows=[dict(**r, ground_truth=answers[r["id"]]) for r in requests[:24]],
    )


if __name__ == "__main__":
    save("research/data/tool-bfcl-math-01.json", material())
