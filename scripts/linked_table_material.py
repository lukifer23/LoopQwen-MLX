"""Render full source files and exercise real one- and two-lookup paths for open-answer questions."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen import qwen_tools
from loopqwen.csv_tools import tools as csv_tools
from loopqwen.identity import fingerprint
from loopqwen.io import read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.reasoned_evidence_data import table
from scripts.linked_table_archive import Z
from scripts.linked_table_sources import DEST, contains, words

FILES = ("records.csv", "articles.csv")
REPORT = Path("docs/measurements/linked-table-callbacks.json")


def cases():
    enrolled = read(DEST / "enrollment.json")["items"]
    index = {w["question_id"]: w for group in enrolled.values() for w in group}
    for line in gzip.decompress(Z.read_bytes()).splitlines():
        original = json.loads(line)
        yield (
            original["split"],
            index[original["annotation"]["question_id"]],
            json.loads(original["tables_tok"]),
            json.loads(original["request_tok"]),
        )


def documents(source, pages):
    # Each original text field is preserved separately from its original linked page IDs.
    # A positional prefix keeps headers unique for tables with repeated or empty names.
    names = [f"{i + 1}. {v[0] or 'Untitled'}" for i, v in enumerate(source["header"])]
    columns = [label for n in names for label in (n, n + " [links]")]
    records = [
        [v for value, links in row for v in (value, " | ".join(links))]
        for row in source["data"]
    ]
    return names, {
        FILES[0]: table(columns, records),
        FILES[1]: table(["Page", "Article"], list(pages.items())),
    }


def question(w, source, mode, records):
    if mode == "inline":
        intro = f"The table {source['title']} is below. The attached articles.csv contains the text of each linked page.\n\n{records}"
    else:
        intro = f"The attached records.csv contains the table {source['title']} and its links. The attached articles.csv contains the text of each linked page."
    return f"{intro}\n\nUse the supplied evidence to answer this question with a short answer.\n{w['question']}"


def oracle(w, source, pages, names, mode, tool):
    arc = source["data"][w["row"]]
    i = w["column"]
    eligible = [
        j
        for j, (value, _) in enumerate(arc)
        if j != i
        and [source["header"][j][0], value] in w["cue_cells"]
        and sum(row[j][0] == value for row in source["data"]) == 1
    ]
    key = max(eligible, key=lambda j: (len(arc[j][0]), -j))
    args = [
        {
            "document": FILES[0],
            "key_column": names[key],
            "key": arc[key][0],
            "value_column": names[i] + " [links]",
        },
        {
            "document": FILES[1],
            "key_column": "Page",
            "key": w["link"],
            "value_column": "Article",
        },
    ]
    events = []
    for a in args if mode == "file" else args[1:]:
        obs = tool.callback(**a)
        events.append({"call": {"name": tool.name, "arguments": a}, "observation": obs})
    if mode == "file":
        assert w["link"] in events[0]["observation"]["value"].split(" | ")
    assert events[-1]["observation"] == {"value": pages[w["link"]]} and contains(
        words(pages[w["link"]]), words(w["answer"])
    )
    return events


def render(w, source, pages, folder):
    names, docs = documents(source, pages)
    for name, s in docs.items():
        (folder / name).write_bytes(s.encode())
    result = {}
    for mode, available in (("inline", FILES[1:]), ("file", FILES)):
        provided = csv_tools([folder / n for n in available], keyed=True)
        expected = oracle(w, source, pages, names, mode, provided[0])
        prompt = question(w, source, mode, docs[FILES[0]])
        result[mode] = {
            "question": prompt,
            "events": expected,
            "definitions": qwen_tools.declarations(provided),
            "files": list(available),
        }
    return result


def run():
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    material = []
    lengths = {}
    counts = Counter()
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for split, w, source, pages in cases():
            for mode, v in render(w, source, pages, folder).items():
                msgs = [{"role": "user", "content": v["question"]}]
                ids = tok.encode(
                    qwen_tools.render(tok, msgs, v["definitions"]),
                    add_special_tokens=False,
                )
                key = split + "." + mode
                counts[key] += len(v["events"])
                lengths.setdefault(key, []).append(len(ids))
                material.append(
                    {
                        "id": w["question_id"] + "-" + mode,
                        "prompt": fingerprint(ids),
                        "callback_sequence": fingerprint(v["events"]),
                    }
                )
    receipt = {
        "enrollment": fingerprint(read(DEST / "enrollment.json")),
        "request_count": len(material),
        "actual_callbacks": dict(counts),
        "prompt_tokens": {
            k: {"minimum": min(v), "maximum": max(v), "total": sum(v)}
            for k, v in lengths.items()
        },
        "items": material,
    }
    assert (len(material), sum(counts.values())) == (304, 456)
    if REPORT.exists():
        assert read(REPORT) == receipt
    else:
        save(REPORT, receipt)
    print(
        "Verified",
        len(material),
        "ordinary reasoning-enabled prompt streams and",
        sum(counts.values()),
        "unmodified public CSV callbacks.",
    )


def example(ident, dest):
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise ValueError("The example export directory must be empty")
    for _, w, source, articles in cases():
        if w["question_id"] == ident:
            for mode, v in render(w, source, articles, dest).items():
                (dest / (mode + "-question.txt")).write_bytes(v["question"].encode())
            return
    raise ValueError("Unknown question identifier")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--example")
    p.add_argument("--directory", type=Path)
    args = p.parse_args()
    if bool(args.example) != bool(args.directory):
        p.error("Use --example and --directory together")
    if args.example:
        example(args.example, args.directory)
    else:
        run()
