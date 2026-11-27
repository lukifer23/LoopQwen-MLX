"""Measure the isolated CPU cost of reconstructing a complete native tool prompt trace."""

import gzip
from time import perf_counter, process_time

from loopqwen.file_tools import definitions
from loopqwen.identity import fingerprint
from loopqwen.io import read, save
from loopqwen.qwen_tools import declarations, render
from loopqwen.research.cross_view_study import tokenizer
from scripts.locked_context_material import SOURCE, VERSIONS, request


def run():
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    panel = read("research/data/locked-context.json")["rows"]
    rounds = []
    for rev, row in zip(VERSIONS, panel, strict=True):
        text = request(
            gzip.decompress((SOURCE / (rev[0] + ".lock.gz")).read_bytes()).decode()
        )
        chat = render(
            tok,
            [{"role": "user", "content": text}],
            declarations(definitions(None, ["uv.lock"])),
        )
        actual = tok.encode(chat, add_special_tokens=False)
        assert fingerprint(actual) == row["prompt_sha256"]
        for group in range(4):
            for mode in ["reuse", "encode"] if group % 2 == 0 else ["encode", "reuse"]:
                start = perf_counter()
                cpu = process_time()
                traced = (
                    actual
                    if mode == "reuse"
                    else tok.encode(chat, add_special_tokens=False)
                )
                wall = perf_counter() - start
                cputime = process_time() - cpu
                assert traced == actual
                rounds.append(
                    {
                        "git": rev[0],
                        "round": group + 1,
                        "mode": mode,
                        "postdecode_s": wall,
                        "cpu_s": cputime,
                        "ids": len(actual),
                    }
                )
    save(
        "docs/measurements/tool-trace-input-cost.json",
        {
            "scope": "CPU-only cost of recording the ordinary prompt ID sequence after native generation. The references are previously computed, complete IDs from the same tokenizer; no decoder latency or model-quality claim.",
            "inputs": "research/data/locked-context.json",
            "rounds": rounds,
        },
    )
    print([(r["mode"], round(r["postdecode_s"], 4)) for r in rounds])


if __name__ == "__main__":
    run()
