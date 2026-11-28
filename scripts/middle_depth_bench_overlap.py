"""Audit the public enrollment for shared sources and exact adaptation-text overlap."""

import gzip
import json
import re
from zipfile import ZipFile
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from scripts.middle_depth_bench_sources import DATA, OUT


def words(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def audit():
    tokenizer = Tokenizer.from_file("exports/standalone-middle-576/base/tokenizer.json")
    training = Path("research/sources/middle-depth/training.json.gz")
    with gzip.open(training) as stream:
        bank = [
            (r["id"], words(tokenizer.decode(r["p"] + r["y"])))
            for r in json.load(stream)
        ]
    archive = Path("data/research_external/winogrande/source.zip")
    with ZipFile(archive) as zipped:
        with zipped.open("winogrande_1.1/dev.jsonl") as file:
            originals = {r["sentence"]: r for r in map(json.loads, file)}
    clusters = {}
    matches = []
    for name, spec in read(OUT / "enrollment.json")["tasks"].items():
        subset = (
            pq.read_table(DATA / (name + ".parquet")).take(spec["indices"]).to_pylist()
        )
        groups = {}
        for index, doc in zip(spec["indices"], subset, strict=True):
            if name == "winogrande":
                orig = originals[doc["sentence"]]
                if any(orig[k] != v for k, v in doc.items()):
                    raise ValueError("Original WinoGrande row differs")
                group = orig["qID"].rsplit("-", 1)[0]
                question = doc["sentence"]
            elif name == "hellaswag":
                group, question = doc["source_id"], doc["ctx"]
            elif name == "sciq":
                group, question = doc["support"].strip() or str(index), doc["question"]
            else:
                group = doc["id"]
                question = doc[
                    "question" if name == "arc_challenge" else "question_stem"
                ]
            groups[str(index)] = fingerprint(group)
            for row, text in bank:
                if " " + words(question) + " " in " " + text + " ":
                    options = doc.get("choices", {}).get("text", [])
                    matches.append(
                        {
                            "task": name,
                            "index": index,
                            "training_row": row,
                            "stem_words": len(words(question).split()),
                            "option_text_hits": sum(
                                " " + words(x) + " " in " " + text + " "
                                for x in options
                            ),
                        }
                    )
        clusters[name] = groups
    save(
        OUT / "source-audit.json",
        {
            "enrollment": digest(OUT / "enrollment.json"),
            "training_rows": len(bank),
            "training_sha": digest(training),
            "winogrande_original": digest(archive),
            "method": "Normalized whole-word question text in decoded adaptation prompts and targets. Cluster by original WinoGrande pair, HellaSwag source, nonempty SciQ passage, or ARC and OpenBookQA item ID. Empty SciQ supports are separate items. Pretraining and paraphrase overlap are unknown.",
            "matches": matches,
            "clusters": clusters,
        },
    )


if __name__ == "__main__":
    audit()
