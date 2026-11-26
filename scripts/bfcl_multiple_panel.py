"""Select an external multiple-function panel by fixed order and declared interface."""

import hashlib
import json
import re
from pathlib import Path

from huggingface_hub import hf_hub_download
from jsonschema import Draft202012Validator, SchemaError

from loopqwen.io import digest, read, save


def select(seed="external-multi-41192", excluded=(), additional_training=()):
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
    seen = set(excluded)
    from loopqwen.tool_session import SYSTEM

    for r in [
        *read("research/data/tool-dialogue-typed-training.json")["examples"],
        *additional_training,
    ]:
        seen.update(
            f["name"]
            for f in json.loads(r["messages"][0]["content"].removeprefix(SYSTEM))
        )
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
        key=lambda x: hashlib.sha256((seed + x["id"]).encode()).digest(),
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
    return dict(
        repository=REPO, revision=REV, sha256=[digest(f) for f in files], rows=rows
    )


if __name__ == "__main__":
    import argparse
    import random

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterfactual-retention", action="store_true")
    if parser.parse_args().counterfactual_retention:
        old = []
        for label, file in [
            ("simple", "tool-bfcl-math-01"),
            ("multiple", "tool-bfcl-multiple-01"),
        ]:
            old.extend(
                dict(w, panel=label)
                for w in read("research/data/" + file + ".json")["rows"]
            )
        used = {f["name"] for w in old for f in w["function"]}
        material = select(
            "retention-multi-b2-175027",
            used,
            read("research/data/tool-counterfactual-training.json")["examples"],
        )
        rows = old + [dict(w, panel="fresh") for w in material["rows"]]
        random.Random(32198).shuffle(rows)
        assert len({w["id"] for w in rows}) == 72
        material["rows"] = rows
        save("research/data/tool-counterfactual-bfcl.json", material)
    else:
        material = select()
        save("research/data/tool-bfcl-multiple-01.json", material)
    print([r["id"] for r in material["rows"]])
