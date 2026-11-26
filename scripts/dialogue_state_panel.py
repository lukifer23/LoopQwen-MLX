"""Freeze missing and complete inputs to real CSV and time-zone operations."""

import hashlib
import random
import subprocess
from pathlib import Path

from mlx_lm.utils import load_tokenizer

from loopqwen.io import read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.research.action_preferences import documents
from loopqwen.resource_tools import tools as resource_tools
from loopqwen.tool_session import initial_messages, prompt

REV = "447a795"
DATA = Path("research/data")
ZONES = [
    ("America/St_Johns", "2030-01-16", 0, "zone"),
    ("Europe/Paris", "2031-07-03", 1, "date"),
    ("Asia/Kathmandu", "2029-11-21", 0, "date"),
    ("Australia/Adelaide", "2032-05-19", 0, "zone"),
    ("Pacific/Chatham", "2030-02-07", 1, "zone"),
    ("America/Toronto", "2030-09-10", 1, "date"),
    ("Europe/London", "2029-12-14", 0, "zone"),
    ("Australia/Lord_Howe", "2028-01-23", 1, "date"),
]
CSVS = [
    ("folded-system.csv", 1, "success", "row"),
    ("folded-system.csv", 5, "completed", "column"),
    ("folded-system.csv", 0, "", "document"),
    ("folded-system.csv", 8, "n", "row"),
    ("draft-program-requests.csv", 7, "gold", "column"),
    ("draft-program-requests.csv", 11, "output_tokens", "row"),
    ("draft-program-requests.csv", 0, "", "document"),
    ("draft-program-requests.csv", 21, "draft_length", "column"),
]


def main():
    tok = load_tokenizer(local_snapshot())
    prior = read(read("research/experiments/action-preference-01.json")["material"])
    m = {"sheets": {}, "rows": [w for w in prior["rows"] if w["kind"] == "public"]}
    for filename in dict.fromkeys(c[0] for c in CSVS):
        payload = subprocess.check_output(["git", "show", f"{REV}:paper/{filename}"])
        m["sheets"][filename] = {
            "text": payload.decode(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "source": REV + ":paper/" + filename,
        }

    def enroll(i, domain, kind, full, gap, detail, fn, args, tools):
        for label, users in (("complete", [full]), ("missing", [gap, detail])):
            m["rows"].append(
                {
                    "id": f"state-live-{i}-{label}",
                    "anchor": i,
                    "domain": domain,
                    "kind": label,
                    "missing": kind,
                    "users": users,
                    "call": {"name": fn.name, "arguments": args},
                    "gold": fn.callback(**args)["value"],
                    "initial_ids": tok.encode(
                        prompt(tok, initial_messages(users[0], tools)),
                        add_special_tokens=False,
                    ),
                }
            )

    with documents(m) as files:
        for i, (file, row, col, gap) in enumerate(CSVS):
            if row:
                full = (
                    f"Please return the {col} field from data record {row} in {file}."
                )
                missing = (
                    f"Please return the {col} field from a data record in {file}."
                    if gap == "row"
                    else f"I need the value of a field in record {row} of {file}."
                )
                detail = (
                    f"Use record {row}."
                    if gap == "row"
                    else f"The column I need is {col}."
                )
                args = {"document": file, "row": row, "column": col}
                fn = files[1]
            else:
                full = f"Count the records below the header in {file}."
                missing = "I need the record count in one of my CSV files."
                detail = f"Use {file}."
                args = {"document": file}
                fn = files[0]
            enroll(i, "files", gap, full, missing, detail, fn, args, files)
    tools = resource_tools("zone")
    for i, (zone, date, op, gap) in enumerate(ZONES, 8):
        subject = (
            "total offset from UTC in minutes",
            "daylight-saving contribution in minutes",
        )[op]
        full = f"Look up the {subject} at local noon in {zone} on {date}."
        missing = (
            f"Look up the {subject} at local noon on {date}."
            if gap == "zone"
            else f"Look up the {subject} at local noon in {zone}."
        )
        detail = (
            f"The time zone is {zone}."
            if gap == "zone"
            else f"The date to look up is {date}."
        )
        enroll(
            i,
            "zones",
            gap,
            full,
            missing,
            detail,
            tools[op],
            {"zone": zone, "date": date},
            tools,
        )
    random.Random(9384701).shuffle(m["rows"])
    save(DATA / "dialogue-state-panel.json", m)


if __name__ == "__main__":
    main()
