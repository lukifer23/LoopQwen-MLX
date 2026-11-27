"""Align real human source citations to causal query positions without altering model inputs."""

import json
from collections import Counter
from zipfile import ZipFile

from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from scripts.shared_query_material import ROOT, paragraph_ids, source


def offsets(tok, ids):
    text = tok.decode(ids)
    encoded = tok._tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True
    )
    assert encoded["input_ids"] == ids
    return text, encoded["offset_mapping"]


def token_ranges(text, spans, offsets):
    values = [
        i
        for i, (a, b) in enumerate(offsets)
        if text[a:b].strip() and any(a < j and b > k for k, j in spans)
    ]
    ranges = []
    for i in values:
        if ranges and ranges[-1][1] == i:
            ranges[-1][1] = i + 1
        else:
            ranges.append([i, i + 1])
    if not ranges:
        raise ValueError("Empty source annotation")
    return ranges


def finance_spans(raw):
    pre = raw["pre_text"]
    post = raw["post_text"]
    parts = [*pre, *[" | ".join(r) for r in raw["table"]], *post]
    offsets = []
    current = 0
    for v in parts:
        offsets.append((current, current + len(v)))
        current += len(v) + 1
    spans = []
    for key in raw["qa"]["gold_inds"]:
        kind, num = key.split("_")
        i = int(num)
        if kind == "table":
            if not 0 <= i < len(raw["table"]):
                raise ValueError("Invalid cited table row")
            i += len(pre)
        elif kind == "text":
            if not 0 <= i < len(pre) + len(post):
                raise ValueError("Invalid cited prose index")
            if i >= len(pre):
                i += len(raw["table"])
        else:
            raise ValueError("Invalid financial source index")
        spans.append(offsets[i])
    return "\n".join(parts), spans


def build():
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    with ZipFile(source("strategyqa/strategyqa_dataset.zip")) as z:
        strategy = {r["qid"]: r for r in json.loads(z.read("strategyqa_train.json"))}
        paragraphs = json.loads(z.read("strategyqa_train_paragraphs.json"))
    fin = {r["id"]: r for r in read(source("finqa/train.json"))}
    out = {}
    counts = Counter()
    for w in unpack(ROOT / "training.json.gz"):
        g = w["group"]
        if g not in ("strategy", "report_action"):
            continue
        p, y = token_pair(tok, w)
        src, p_off = offsets(tok, p)
        target, t_off = offsets(tok, y)
        user = w["messages"][0]["content"]
        start = src.index(user)
        labels = []
        if g == "report_action":
            r = fin[w["id"].removeprefix("finqa/").removesuffix("/action")]
            text, spans = finance_spans(r)
            assert user == text + "\n\n" + r["qa"]["question"]
            labels.append(
                dict(
                    query=0,
                    source=token_ranges(
                        src, [(start + a, start + b) for a, b in spans], p_off
                    ),
                )
            )
        else:
            r = strategy[w["id"].removeprefix("strategyqa/")]
            annotations = []
            for trace in r["evidence"]:
                if len(trace) != len(r["decomposition"]) or not all(
                    paragraph_ids(step, paragraphs) or "operation" in step
                    for step in trace
                ):
                    continue
                keys = list(dict.fromkeys(paragraph_ids(trace, paragraphs)))
                blocks = [
                    paragraphs[k]["title"] + "\n" + paragraphs[k]["content"]
                    for k in keys
                ]
                if "\n\n".join([*blocks, r["question"].strip()]) == user:
                    annotations.append(trace)
            assert annotations, w["id"]
            trace = annotations[0]
            for i, (subq, step) in enumerate(
                zip(r["decomposition"], trace, strict=True), 1
            ):
                keys = list(dict.fromkeys(paragraph_ids(step, paragraphs)))
                if not keys:
                    continue
                spans = []
                for k in keys:
                    block = paragraphs[k]["title"] + "\n" + paragraphs[k]["content"]
                    a = user.index(block)
                    spans.append((start + a, start + a + len(block)))
                line = f"{i}. {subq}"
                line = line.rstrip()
                if line not in target:
                    raise ValueError(
                        f"Could not locate annotated subquestion in {w['id']}"
                    )
                char = target.index(line)
                q = next(j for j, (a, b) in enumerate(t_off) if a <= char < b)
                labels.append(dict(query=q, source=token_ranges(src, spans, p_off)))
        assert labels and all(
            0 <= x["query"] < len(y)
            and all(0 <= a < b <= len(p) for a, b in x["source"])
            for x in labels
        )
        out[w["id"]] = dict(
            prefix=fingerprint(p), labels=fingerprint(y), evidence=labels
        )
        counts[g] += len(labels)
    packed(ROOT / "evidence.json.gz", out)
    save(
        ROOT / "evidence-inventory.json",
        dict(
            traces=len(out),
            queries=dict(counts),
            training=digest(ROOT / "training.json.gz"),
            evidence=digest(ROOT / "evidence.json.gz"),
        ),
    )
    print(len(out), "original sources,", dict(counts), "human-cited causal queries")


if __name__ == "__main__":
    build()
