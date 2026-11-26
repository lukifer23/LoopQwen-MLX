"""Freeze human single-passage questions from independent publisher splits."""

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

from mlx_lm.utils import load_tokenizer

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.passage_copy import request

ROOT = Path("data/research_external")
SOURCES = {
    ("SQuAD", "train"): (
        "squad-train-v1.1.json",
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
        "3527663986b8295af4f7fcdff1ba1ff3f72d07d61a20f487cb238a6ef92fd955",
    ),
    ("SQuAD", "dev"): (
        "squad-dev-v1.1.json",
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
        "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9",
    ),
    ("QuAC", "train"): (
        "quac-train-v0.2.json",
        "https://s3.amazonaws.com/my89public/quac/train_v0.2.json",
        "ff5cca5a2e4b4d1cb5b5ced68b9fce88394ef6d93117426d6d4baafbcc05c56a",
    ),
    ("QuAC", "dev"): (
        "quac-val-v0.2.json",
        "https://s3.amazonaws.com/my89public/quac/val_v0.2.json",
        "09e622916280ba04c9352acb1bc5bbe80f11a2598f6f34e934c51d9e6570f378",
    ),
}


def normalized(s):
    return " ".join(re.findall(r"[^\W_]+", s.casefold()))


def main():
    tok = load_tokenizer(local_snapshot())
    used_titles = {
        normalized(t)
        for n in ("natural-context-01", "prose-reading-01")
        for t in read(f"research/experiments/{n}.json")["titles"]
    }
    used_titles.update(
        normalized(r["source"])
        for r in read("research/data/conditional-language-02.json")
        if r["family"] == "reading"
    )
    seen = set()
    paths = {}
    for split in ("dev", "train"):
        enrolled = []
        for domain in ("SQuAD", "QuAC"):
            file, url, expected = SOURCES[domain, split]
            path = ROOT / file
            if not path.is_file():
                data = urlopen(url, timeout=90).read()
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError("Source digest mismatch")
                path.write_bytes(data)
            if digest(path) != expected:
                raise ValueError("Source digest mismatch")
            paths[file] = {"url": url, "sha256": expected}
            pool = []
            for article in json.loads(path.read_bytes())["data"]:
                title = article["title"].replace("_", " ").strip()
                if normalized(title) in used_titles:
                    continue
                for p in article["paragraphs"]:
                    context = p["context"].removesuffix(" CANNOTANSWER").strip()
                    for q in p["qas"][:1] if domain == "QuAC" else p["qas"]:
                        pool.append((title, context, q))
            random.Random(f"passage-copy-71731-{domain}-{split}").shuffle(pool)
            n = 32 if split == "dev" else 1024
            grouped = Counter()
            chosen = []
            strata = Counter()
            for title, passage, q in pool:
                identity = hashlib.sha256(normalized(passage).encode()).hexdigest()
                if identity in seen or grouped[title] >= (
                    4
                    if split == "dev" and domain == "SQuAD"
                    else 1
                    if split == "dev"
                    else 32
                ):
                    continue
                annotations = q["answers"]
                null = sum(a["text"] == "CANNOTANSWER" for a in annotations) * 2 >= len(
                    annotations
                )
                if (
                    domain == "QuAC"
                    and split == "dev"
                    and strata[null] >= (8 if null else 24)
                ):
                    continue
                references = (
                    ["Cannot answer"]
                    if null
                    else list(
                        dict.fromkeys(
                            a["text"]
                            for a in annotations
                            if a["text"] != "CANNOTANSWER"
                        )
                    )
                )
                question = q["question"].strip()
                _text, ids, span, _, _ = request(tok, title, passage, question)
                # Only constrain training activation size. Every frozen dev passage
                # is retained in full, including unanswerable and long responses.
                annotations = q["answers"]
                preferred = q.get("orig_answer", annotations[0])
                answer = preferred["text"]
                if answer == "CANNOTANSWER":
                    answer = "Cannot answer"
                reply = tok.encode(
                    "</think>\n\n" + answer, add_special_tokens=False
                ) + [tok.convert_tokens_to_ids("<|im_end|>")]
                if split == "train" and len(ids) + len(reply) > 512:
                    continue
                raw = q.get("orig_answer", annotations[0])
                row = {
                    "id": q["id"],
                    "domain": domain,
                    "title": title,
                    "question": question,
                    "passage": passage,
                    "references": references,
                    "unanswerable": null,
                    "answer": answer,
                    "answer_char": raw["answer_start"],
                    "sha256": fingerprint(ids),
                    "source_span": list(span),
                }
                if split == "dev":
                    row["prompt_ids"] = ids
                seen.add(identity)
                grouped[title] += 1
                strata[null] += 1
                chosen.append(row)
                if len(chosen) == n:
                    break
            if len(chosen) != n:
                raise ValueError((domain, split, len(chosen)))
            enrolled.extend(chosen)
            print(domain, split, len(chosen), len(grouped), flush=True)
        if split == "dev":
            used_titles.update(normalized(r["title"]) for r in enrolled)
        random.Random("read-crossover-83724-" + split).shuffle(enrolled)
        save(
            f"research/data/passage-copy-{split}.json",
            {"sources": dict(paths), "rows": enrolled},
        )


if __name__ == "__main__":
    main()
