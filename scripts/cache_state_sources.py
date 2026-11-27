"""Freeze eight real, previously unmeasured training prompts for a same-weight timing crossover."""

from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen import qwen_tools
from loopqwen.identity import fingerprint
from loopqwen.io import read, save
from loopqwen.research.cross_view_study import tokenizer
from scripts.linked_table_material import REPORT, cases, render

DEST = Path("research/data/cache-state-prompts.json")


def run():
    all_rows = read(REPORT)["items"]
    train = {w["question_id"] for split, w, _, _ in cases() if split == "train"}
    pool = sorted(
        [v for v in all_rows if v["id"].rsplit("-", 1)[0] in train],
        key=lambda v: fingerprint([891211, v["id"]]),
    )
    ids = {
        v["id"]
        for mode in ("file", "inline")
        for v in [w for w in pool if w["id"].endswith("-" + mode)][:4]
    }
    rows = []
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    with TemporaryDirectory() as tmp:
        for split, w, s, docs in cases():
            if not any(w["question_id"] + "-" + m in ids for m in ("inline", "file")):
                continue
            assert split == "train"
            for mode, v in render(w, s, docs, Path(tmp)).items():
                rid = w["question_id"] + "-" + mode
                if rid not in ids:
                    continue
                msg = [{"role": "user", "content": v["question"]}]
                tokens = tok.encode(
                    qwen_tools.render(tok, msg, v["definitions"]),
                    add_special_tokens=False,
                )
                assert fingerprint(tokens) == next(
                    row["prompt"] for row in pool if row["id"] == rid
                )
                rows.append({"id": rid, "initial_ids": tokens})
    assert len(rows) == 8 and all(len(w["initial_ids"]) > 128 for w in rows)
    data = {"rows": rows}
    if DEST.exists():
        assert read(DEST) == data
    else:
        save(DEST, data)
    print([len(w["initial_ids"]) for w in rows])


if __name__ == "__main__":
    run()
