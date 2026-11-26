"""Select complete human HotpotQA bridge contexts and independent development questions."""

import hashlib
import heapq
import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from mlx_lm.utils import load_tokenizer

from loopqwen.io import digest, save
from loopqwen.model_loader import local_snapshot

REPO = "hotpotqa/hotpot_qa"
REV = "1908d6afbbead072334abe2965f91bd2709910ab"


def canonical(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def enroll(split, count, excluded, tok, salt="bridge-reader-912"):
    file = (
        "distractor/"
        + ("train-00000-of-00002" if split == "train" else "validation-00000-of-00001")
        + ".parquet"
    )
    source = Path(hf_hub_download(REPO, file, repo_type="dataset", revision=REV))
    candidates = []
    for batch in pq.ParquetFile(source).iter_batches(batch_size=64):
        for r in batch.to_pylist():
            ctx = r["context"]
            if (
                r["type"] != "bridge"
                or len(ctx["title"]) != 10
                or len(set(ctx["title"])) != 10
                or r["answer"].casefold() in ("yes", "no")
            ):
                continue
            full = ["".join(ss).strip() for ss in ctx["sentences"]]
            if (
                not all(full)
                or sum(map(len, full)) > 6800
                or max(map(len, full)) > 1600
            ):
                continue
            titles = list(dict.fromkeys(r["supporting_facts"]["title"]))
            if len(titles) != 2 or any(
                t not in ctx["title"] or canonical(t) in excluded for t in titles
            ):
                continue
            if canonical(r["question"]) in excluded:
                continue
            indices = [ctx["title"].index(t) for t in titles]
            # Gold is used to order training hop labels only. The held-out endpoint uses unordered supports.
            if split == "train":
                found = [
                    i
                    for i in indices
                    if " " + canonical(r["answer"]) + " "
                    in " " + canonical(ctx["title"][i] + " " + full[i]) + " "
                ]
                if len(found) != 1:
                    continue
                indices = [next(i for i in indices if i != found[0]), found[0]]
            priority = int.from_bytes(
                hashlib.sha256((salt + ":" + r["id"]).encode()).digest(),
                "big",
            )
            if len(candidates) >= count * 6 and priority >= -candidates[0][0]:
                continue
            row = {
                "id": r["id"],
                "question": r["question"],
                "titles": ctx["title"],
                "articles": full,
                "support": indices,
                "gold": r["answer"],
            }
            item = (-priority, r["id"], row)
            if len(candidates) < count * 6:
                heapq.heappush(candidates, item)
            else:
                heapq.heapreplace(candidates, item)
    from loopqwen.bridge_reader import messages

    selected = []
    for _, _, row in sorted(candidates, reverse=True):
        labels = {
            canonical(row["question"]),
            *[canonical(row["titles"][i]) for i in row["support"]],
        }
        if labels & excluded:
            continue
        msgs = messages(row["question"], row["titles"], row["articles"])
        ids = tok.encode(
            tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
            ),
            add_special_tokens=False,
        )
        if len(ids) > 1664:
            continue
        selected.append(
            dict(row, messages=msgs, visible=row["gold"], thinking="")
            if split == "train"
            else row
        )
        excluded.update(labels)
        if len(selected) == count:
            break
    assert len(selected) == count
    return dict(
        source=REPO,
        revision=REV,
        file=file,
        sha256=digest(source),
        **({"examples": selected} if split == "train" else {"rows": selected}),
    )


if __name__ == "__main__":
    t = load_tokenizer(local_snapshot())
    excluded = set()
    for split, n in [("validation", 40), ("train", 384)]:
        data = enroll(split, n, excluded, t)
        save("research/data/bridge-reading-" + split + ".json", data)
        print(split, n, data["sha256"], flush=True)
