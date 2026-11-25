"""Write a frozen reader evaluation prompt as UTF-8 for the native thinking CLI."""

import argparse
import sys
from pathlib import Path

from mlx_lm.utils import load_tokenizer
from loopqwen.io import read
from loopqwen.model_loader import local_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        choices=("attention-memory-01", "attention-memory-128k", "prose-reading-01"),
        default="attention-memory-01",
    )
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    data = read(Path("research/data") / (args.study + ".json"))
    items = data["panel"] if isinstance(data, dict) else data
    if not 0 <= args.index < len(items):
        parser.error(f"Choose a prompt index from 0 through {len(items) - 1}")
    t = load_tokenizer(local_snapshot())
    marker = "USER_CONTENT_HERE"

    def template(user):
        return t.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    prefix, suffix = template(marker).split(marker)
    ids = items[args.index]["prompt_ids"]
    text = t.decode(ids)
    body = text.removeprefix(prefix).removesuffix(suffix)
    if t.encode(template(body), add_special_tokens=False) != ids:
        raise ValueError("Could not reconstruct the frozen thinking request")
    sys.stdout.buffer.write(body.encode("utf-8"))


if __name__ == "__main__":
    main()
