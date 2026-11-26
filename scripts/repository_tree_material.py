"""Reproduce a real whole-Git-tree size lookup at two immutable revisions."""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

from mlx_lm.utils import load_tokenizer

from loopqwen.identity import fingerprint
from loopqwen.io import save
from loopqwen.model_loader import local_snapshot
from loopqwen.tool_session import prompt

REVISIONS = (
    "c6a08ff72dd895ed1578a0c9cc4940159dcc52c8",
    "5bd60c2c678b3f527805e8050f5104a7bc9c78e0",
)
FILE = "paper/loopqwen.pdf"


def snapshot(revision):
    listing = subprocess.check_output(["git", "ls-tree", "-r", "-l", revision])
    prefix = "Here is the complete output of git ls-tree -r -l for this project.\n\n"
    question = f"According to this tree listing, how many bytes are in {FILE}? Give the byte count."
    return listing, prefix + listing.decode() + "\n" + question


def build():
    tok = load_tokenizer(local_snapshot())
    panel = []
    for rev in REVISIONS:
        raw, user = snapshot(rev)
        size = (
            subprocess.check_output(["git", "cat-file", "-s", rev + ":" + FILE])
            .decode()
            .strip()
        )
        text = prompt(tok, [{"role": "user", "content": user}])
        cols = [
            line for line in raw.decode().splitlines() if line.endswith("\t" + FILE)
        ]
        assert len(cols) == 1 and cols[0].split("\t")[0].split()[-1] == size
        lo = text.index(cols[0]) + cols[0].rindex(size + "\t")
        encoded = tok._tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        ids = encoded["input_ids"]
        span = [
            i
            for i, (a, b) in enumerate(encoded["offset_mapping"])
            if b > lo and a < lo + len(size)
        ]
        assert (
            len(ids) >= 131072
            and tok.decode(ids[span[0] : span[-1] + 1]).strip() == size
        )
        panel.append(
            {
                "id": rev,
                "sha256": fingerprint(ids),
                "prompt_ids": ids,
                "listing_sha256": hashlib.sha256(raw).hexdigest(),
                "files": len(raw.splitlines()),
                "path": FILE,
                "gold": size,
                "gold_start": span[0],
                "gold_stop": span[-1] + 1,
            }
        )
    assert panel[0]["gold"] != panel[1]["gold"]
    return {"panel": panel}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-request",
        choices=REVISIONS,
        help="Write the original UTF-8 user message to standard output",
    )
    args = parser.parse_args()
    if args.show_request:
        sys.stdout.write(snapshot(args.show_request)[1])
    else:
        data = build()
        save(Path("research/data/repository-tree-field-01.json"), data)
        for row in data["panel"]:
            print(
                row["id"],
                row["files"],
                len(row["prompt_ids"]),
                row["gold"],
                row["gold_start"],
            )
