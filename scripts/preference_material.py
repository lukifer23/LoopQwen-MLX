"""Freeze external action preferences and matched requests over real unmodified CSV files."""

import argparse
import hashlib
import json
import random
import re
import subprocess
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path

from mlx_lm.utils import load_tokenizer

from loopqwen.csv_tools import tools as csv_tools
from loopqwen.evaluation import executable_tools
from loopqwen.io import read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.tool_session import SYSTEM, compact, initial_messages, prompt

DATA = Path("research/data")
SEED = 373921
TRAIN_HASH = "d90637f108fabf1b097493c5818c3692e1d73140259d2f8e490536255e765bc4"
NVIDIA = "0582f7749df63a96fdc3070932e83e72396ace53"
PENGUINS = "8957207b78d6ccd1b4654a9dd9c9041b657478ab"
PAPER = "4af8d54"
SAFE_TOPIC = re.compile(
    r"weather|sports?|music|books?|recipe|cook|restaurants?|travel|flight|hotels?|animals?|movie|film|artwork|museum|language|basketball|soccer|tennis|geography|location|cities|trains?|bus schedule|ingredients?|tourism|games?|concert|albums?|lyrics",
    re.IGNORECASE,
)
EXCLUDED = re.compile(
    r"credential|password|auth.token|credit.card|exploit|malware|weapon|ammunition|phish|bomb|attack|delete.account|place.order|send.email|transfer.money",
    re.IGNORECASE,
)


def fetch(folder, name, url):
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    if not p.exists():
        p.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    return p.read_bytes()


def convert_response(text):
    if "<TOOLCALL>" not in text:
        return text.strip()
    match = re.fullmatch(r"\s*<TOOLCALL>(.*?)</TOOLCALL>\s*", text, re.DOTALL)
    if not match:
        raise ValueError("Unsupported call syntax")
    calls = json.loads(match[1])
    if len(calls) != 1 or not isinstance(calls[0].get("arguments"), dict):
        raise ValueError("Only single action preferences are used")
    return compact({"name": calls[0]["name"], "arguments": calls[0]["arguments"]})


def prepare_preferences(raw, tok):
    rows = list(enumerate(raw.decode().splitlines()))
    rng = random.Random(SEED)
    rng.shuffle(rows)
    buckets, seen, bank = Counter(), set(), []
    for index, line in rows:
        row = json.loads(line)
        msgs = row["messages"]
        if len(msgs) != 1 or msgs[0]["role"] != "user":
            continue
        q = msgs[0]["content"]
        if not SAFE_TOPIC.search(q) or EXCLUDED.search(line) or q.casefold() in seen:
            continue
        try:
            tools = [json.loads(s) for s in row["tools"]]
            chosen, rejected = (
                convert_response(row[k]["content"])
                for k in ("chosen_response", "rejected_response")
            )
            if (
                not tools
                or chosen == rejected
                or not all(0 < len(tok.encode(s)) <= 80 for s in (chosen, rejected))
            ):
                continue
            modern = []
            aliases = {
                "dict": "object",
                "int": "integer",
                "float": "number",
                "str": "string",
                "bool": "boolean",
                "list": "array",
            }
            for t in tools:
                pars = t["parameters"]
                props = {}
                for k, v in pars.get("properties", {}).items():
                    kind = v.get("type", "string").lower().split(",")[0].strip()
                    kind = aliases.get(kind, kind)
                    if kind not in (
                        "string",
                        "integer",
                        "number",
                        "boolean",
                        "array",
                        "object",
                    ):
                        raise ValueError("Unknown parameter type")
                    props[k] = {
                        "type": kind,
                        "description": str(v.get("description", "")),
                    }
                modern.append(
                    {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": {
                            "type": "object",
                            "properties": props,
                            "required": pars.get("required", t.get("required", [])),
                        },
                    }
                )
            messages = [{"role": "system", "content": SYSTEM + compact(modern)}, *msgs]
            if len(tok.encode(prompt(tok, messages))) > 650:
                continue
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
        group = (
            "call"
            if chosen.startswith('{"name"')
            else "ask"
            if "?" in chosen
            else "other"
        )
        if buckets[group] >= 144:
            continue
        split = "validation" if buckets[group] % 9 == 0 else "train"
        bank.append(
            {
                "source_line": index,
                "group": group,
                "split": split,
                "messages": messages,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
        buckets[group] += 1
        seen.add(q.casefold())
        if len(bank) == 432:
            break
    assert sorted(buckets.values()) == [144, 144, 144], buckets
    return bank


def make(folder):
    tok = load_tokenizer(local_snapshot())
    url = f"https://huggingface.co/datasets/nvidia/When2Call/resolve/{NVIDIA}/train/when2call_train_pref.jsonl"
    raw = fetch(folder, "train.jsonl", url)
    assert hashlib.sha256(raw).hexdigest() == TRAIN_HASH
    prefs = prepare_preferences(raw, tok)
    replay = read(DATA / "tool-dialogue-typed-training.json")["examples"]
    random.Random(SEED).shuffle(replay)
    replay = replay[:192]
    save(
        DATA / "action-preference-training.json",
        {
            "source": url,
            "sha256": TRAIN_HASH,
            "license": "CC BY 4.0",
            "preferences": prefs,
            "replay": replay,
        },
    )
    sheets = {}
    for name in ("penguins.csv", "penguins_raw.csv"):
        src = f"https://raw.githubusercontent.com/allisonhorst/palmerpenguins/{PENGUINS}/inst/extdata/{name}"
        contents = fetch(folder, name, src)
        sheets[name] = {
            "source": src,
            "license": "CC0",
            "sha256": hashlib.sha256(contents).hexdigest(),
            "text": contents.decode(),
        }
    for name in ("mixed-deployment.csv", "matched-tool-effects.csv"):
        contents = subprocess.check_output(["git", "show", f"{PAPER}:paper/{name}"])
        sheets[name] = {
            "source": f"{PAPER}:paper/{name}",
            "sha256": hashlib.sha256(contents).hexdigest(),
            "text": contents.decode(),
        }
    subjects = [
        ("penguins.csv", 12, "body_mass_g", "row"),
        ("penguins.csv", 220, "species", "column"),
        ("penguins.csv", 297, "bill_length_mm", "row"),
        ("penguins.csv", 273, "island", "column"),
        ("penguins_raw.csv", 118, "Individual ID", "row"),
        ("penguins_raw.csv", 51, "Region", "column"),
        ("penguins_raw.csv", 278, "Date Egg", "row"),
        ("penguins_raw.csv", 0, "", "document"),
        ("mixed-deployment.csv", 4, "tokens", "row"),
        ("mixed-deployment.csv", 7, "correct", "column"),
        ("mixed-deployment.csv", 2, "completed", "row"),
        ("mixed-deployment.csv", 0, "", "document"),
        ("matched-tool-effects.csv", 3, "items", "row"),
        ("matched-tool-effects.csv", 5, "clusters", "column"),
        ("matched-tool-effects.csv", 1, "panel", "row"),
        ("matched-tool-effects.csv", 2, "control", "column"),
    ]
    requests = []
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for name, d in sheets.items():
            path = Path(tmp) / name
            path.write_bytes(d["text"].encode())
            files.append(path)
        tools = csv_tools(files)
        for i, (name, row, col, missing) in enumerate(subjects):
            if row:
                query = f"What value is in column {col}, data row {row}, in {name}?"
                incomplete = (
                    f"In {name}, what is the value in column {col}?"
                    if missing == "row"
                    else f"Can you read a field from data row {row} of {name} for me?"
                )
                reply = (
                    f"Data row {row}, please."
                    if missing == "row"
                    else f"The column is {col}."
                )
                args = {"document": name, "row": row, "column": col}
                action = tools[1]
            else:
                query = f"How many data rows does {name} contain?"
                incomplete = [
                    "How many records are there in the file I want to check?",
                    "Can you count the data rows in one of my spreadsheets?",
                ][int(i > 7)]
                reply = f"The file is {name}."
                args = {"document": name}
                action = tools[0]
            for kind, users in (
                ("complete", [query]),
                ("missing", [incomplete, reply]),
            ):
                requests.append(
                    {
                        "id": f"sheet-{i:02}-{kind}",
                        "anchor": i,
                        "domain": "ecology" if i < 8 else "project",
                        "kind": kind,
                        "missing": missing,
                        "users": users,
                        "call": {"name": action.name, "arguments": args},
                        "gold": action.callback(**args)["value"],
                    }
                )
        for r in read(DATA / "tool-exec-bfcl-01.json")["rows"]:
            requests.append(
                {
                    **r,
                    "kind": "public",
                    "users": [
                        r["question"][0][0]["content"] + executable_tools.INSTRUCTION
                    ],
                }
            )
        random.Random(SEED).shuffle(requests)
        for w in requests:
            cat = executable_tools.tools_for(w) if w["kind"] == "public" else tools
            w["initial_ids"] = tok.encode(
                prompt(tok, initial_messages(w["users"][0], cat)),
                add_special_tokens=False,
            )
    save(DATA / "action-preference-panel.json", {"sheets": sheets, "rows": requests})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path)
    make(p.parse_args().cache)
