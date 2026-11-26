"""Select an external multiple-function panel by fixed order and declared interface."""

import hashlib
import json
import re
from pathlib import Path

from huggingface_hub import hf_hub_download
from jsonschema import Draft202012Validator, SchemaError

from loopqwen.io import digest, read, save

if __name__ == "__main__":
    # The source modules also support invocation through python -m scripts...
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.bfcl_panel import REPO, REV
    from scripts.tool_bfcl_study import normalized_definition

    files = [
        Path(hf_hub_download(REPO, name, revision=REV, repo_type="dataset"))
        for name in ("BFCL_v3_multiple.json", "possible_answer/BFCL_v3_multiple.json")
    ]
    questions, labels = [
        [json.loads(line) for line in f.read_text().splitlines()] for f in files
    ]
    labels = {r["id"]: r["ground_truth"] for r in labels}
    seen = set()
    from loopqwen.tool_session import SYSTEM
    for r in read("research/data/tool-dialogue-typed-training.json")["examples"]:
        seen.update(f["name"] for f in json.loads(r["messages"][0]["content"].removeprefix(SYSTEM)))
        if r["visible"].startswith('{"name"'):
            seen.add(json.loads(r["visible"])["name"])
    for r in (
        read("research/data/tool-bfcl-math-01.json")["rows"]
        + read("research/data/tool-exec-bfcl-01.json")["rows"]
    ):
        seen.update(f["name"] for f in r["function"])
    topics = re.compile(
        r"geometr|area|math|theorem|vector|polygon|triangle|circle|route|travel|distance|train|weather|history|museum|paint|sculpt|music|basketball|soccer|baseball|tennis|game|sports|literature|poem|book|mountain|city|geograph|planet|calendar|event|sort|convert|statistics",
        re.I,
    )
    rows = []
    for r in sorted(
        questions,
        key=lambda x: hashlib.sha256(
            ("external-multi-41192" + x["id"]).encode()
        ).digest(),
    ):
        names = {f["name"] for f in r["function"]}
        gold = labels[r["id"]]
        if (
            names & seen
            or len(r["question"][0]) != 1
            or not any(topics.search(n) for opt in gold for n in opt)
        ):
            continue
        if any(
            not isinstance(v, list)
            for opt in gold
            for truth in opt.values()
            for v in truth.values()
        ):
            continue
        try:
            for f in r["function"]:
                Draft202012Validator.check_schema(
                    normalized_definition(f)["parameters"]
                )
        except SchemaError:
            continue
        rows.append(dict(**r, ground_truth=gold))
        if len(rows) == 24:
            break
    assert len(rows) == 24
    save(
        "research/data/tool-bfcl-multiple-01.json",
        dict(
            repository=REPO, revision=REV, sha256=[digest(f) for f in files], rows=rows
        ),
    )
    print([r["id"] for r in rows])
