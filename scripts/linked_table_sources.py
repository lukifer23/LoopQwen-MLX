"""Enroll original open-answer human questions with a verifiable table-to-passage link."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from loopqwen.io import digest, read, save

CACHE = Path("data/research_external/hybridqa")
QUESTIONS = "db22fda8c5951438fade3c69d75b350335ba93b3"
WIKITABLES = "dc066e1a6d5281511d8b73a6107d5ad2824cc2b2"
TRACED = {
    "train": "5008fc2e4b8859f729201fe4a45800dba95686ab5cde5a9f1a66b5794e27fd5b",
    "dev": "2fe4ca01e23d4e589ee6a79f46d7eeeddefd1d33a581c7ceb8955902da3f19c1",
}
TREES = {
    "tables_tok": "772e97624b08e0bd5534b8991a8c2dd8fb6b153f",
    "request_tok": "f1c1a1dab31b2341f338a6a7bf626b5b8c569f20",
}
DEST = Path("research/sources/linked-tables")


def download(url, path, acquire):
    if path.exists():
        return path.read_bytes()
    if not acquire:
        raise FileNotFoundError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=45) as response:
        raw = response.read()
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return raw


def tree_members(kind, acquire):
    tree = json.loads(
        download(
            f"https://api.github.com/repos/wenhuchen/WikiTables-WithLinks/git/trees/{TREES[kind]}",
            CACHE / (kind + "-tree.json"),
            acquire,
        )
    )
    assert tree["sha"] == TREES[kind] and not tree["truncated"]
    return {v["path"]: v["sha"] for v in tree["tree"]}


def blobs(table, trees, acquire):
    out = {}
    for kind in TREES:
        name = table + ".json"
        raw = download(
            f"https://raw.githubusercontent.com/wenhuchen/WikiTables-WithLinks/{WIKITABLES}/{kind}/{quote(name, safe='')}",
            CACHE / kind / name,
            acquire,
        )
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        assert trees[kind][name] == blob
        out[kind] = json.loads(raw)
    return out


def words(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def contains(text, phrase):
    return " " + phrase + " " in " " + text + " "


def verify_case(w, archive, hashes):
    t, docs = archive["tables_tok"], archive["request_tok"]
    answer = words(w["answer-text"])
    q = words(w["question"])
    if not 2 <= len(answer) <= 60 or contains(q, answer):
        return None
    _, (row, col), link, _ = w["answer-node"][0]
    cells = t["data"]
    if not (
        3 <= len(cells) <= 28
        and 2 <= len(t["header"]) <= 8
        and all(len(r) == len(t["header"]) for r in cells)
    ):
        return None
    if link not in cells[row][col][1] or link not in docs:
        return None
    if contains(q, words(link.removeprefix("/wiki/").replace("_", " "))) or contains(
        q, words(cells[row][col][0])
    ):
        return None
    if any(contains(words(c[0]), answer) for r in cells for c in r):
        return None
    actual = {key for key, text in docs.items() if contains(words(text), answer)}
    if actual != {link} or len(docs) < 3:
        return None
    # At least one *other* visible field identifies the needed row in the human question.
    # This check applies only to reader enrollment and creates no inference-time hint.
    candidate = []
    for j, (value, _) in enumerate(cells[row]):
        cue = words(value)
        if j == col or len(cue) < 3 or not contains(q, cue):
            continue
        if sum(words(r[j][0]) == cue for r in cells) == 1:
            candidate.append([t["header"][j][0], value])
    if not candidate:
        return None
    return {
        "question_id": w["question_id"],
        "question": w["question"],
        "answer": w["answer-text"],
        "table": w["table_id"],
        "table_url": t["url"],
        "table_title": t["title"],
        "rows": len(cells),
        "columns": len(t["header"]),
        "linked_documents": len(docs),
        "row": row,
        "column": col,
        "link": link,
        "cell": cells[row][col][0],
        "cue_cells": candidate,
        "table_sha256": hashes["tables_tok"],
        "passages_sha256": hashes["request_tok"],
    }


def run(acquire=False):
    trees = {k: tree_members(k, acquire) for k in TREES}
    selected = {}
    forbidden = set()
    reserved_targets = set()
    reserved_documents = set()
    for split, limit in (("dev", 24), ("train", 128)):
        path = CACHE / (split + ".traced.json")
        raw = download(
            f"https://raw.githubusercontent.com/wenhuchen/HybridQA/{QUESTIONS}/released_data/{split}.traced.json",
            path,
            acquire,
        )
        assert hashlib.sha256(raw).hexdigest() == TRACED[split]
        groups = defaultdict(list)
        for w in json.loads(raw):
            node = w["answer-node"]
            if (
                len(node) == 1
                and node[0][-1] == "passage"
                and w["table_id"] + ".json" in trees["tables_tok"]
            ):
                groups[w["table_id"]].append(w)

        # Sort once by source identity, independent of any fitted model or measure.
        def rank(k):
            return hashlib.sha256(("linked-source-61271/" + k).encode()).digest()

        tables = sorted(groups, key=rank)
        selected[split] = []
        examined = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            for batch in range(0, len(tables), 8):
                ids = tables[batch : batch + 8]
                for table, src in zip(
                    ids, pool.map(lambda i: blobs(i, trees, acquire), ids), strict=True
                ):
                    examined += 1
                    if split == "train" and reserved_targets.intersection(
                        src["request_tok"]
                    ):
                        continue
                    for w in sorted(
                        groups[table], key=lambda v: rank(v["question_id"])
                    ):
                        case = verify_case(
                            w,
                            src,
                            {k: digest(CACHE / k / (table + ".json")) for k in TREES},
                        )
                        if case is None:
                            continue
                        identities = {
                            case["table_url"],
                            words(case["table_title"]),
                            case["link"],
                            words(case["question"]),
                        }
                        if identities & forbidden or (
                            split == "train" and case["link"] in reserved_documents
                        ):
                            continue
                        forbidden.update(identities)
                        if split == "dev":
                            reserved_targets.add(case["link"])
                            reserved_documents.update(src["request_tok"])
                        selected[split].append(case)
                        break
                    if len(selected[split]) == limit:
                        break
                if len(selected[split]) == limit:
                    break
        assert len(selected[split]) == limit
        print(
            split,
            len(selected[split]),
            "distinct human questions, inspected",
            examined,
            "original tables",
            flush=True,
        )
    DEST.mkdir(parents=True, exist_ok=True)
    dest = DEST / "enrollment.json"
    summary = {
        "questions_revision": QUESTIONS,
        "tables_revision": WIKITABLES,
        "question_archives": TRACED,
        "trace_origin": "Authors' automatic string matching; human labels are the original questions and answers. Enrollment additionally checks a unique cross-column row cue, the live cell link, and the unique answer-bearing passage.",
        "items": selected,
    }
    if dest.exists():
        assert read(dest) == summary
    else:
        save(dest, summary)
    bysplit = {s: Counter(w["table"] for w in rows) for s, rows in selected.items()}
    assert not set(bysplit["train"]) & set(bysplit["dev"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire", action="store_true")
    run(parser.parse_args().acquire)
