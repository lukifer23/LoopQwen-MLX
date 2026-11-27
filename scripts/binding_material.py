"""Build disjoint paired human questions from original WinoGrande and WikiSQL rows."""

import csv
import hashlib
import io
import itertools
import json
import re
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from interpolation_material import hashed
from mlx_lm.utils import load_tokenizer
from table_question_panel import SHA as SQL_SHA
from table_question_panel import URL as SQL_URL

from loopqwen import qwen_tools
from loopqwen.csv_tools import tools
from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.tool_session import compact

DATA = Path("research/data")
WIN_URL = (
    "https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip"
)
WIN_SHA = "3619ab104d8be2977b25c90ff420cb42d491707dcc75362a1e5d22bc082b7318"
TRAIN = DATA / "role-binding-training.json"
PANEL = DATA / "role-binding-panel.json"


def fetch(rel, url, sha):
    path = Path("data/research_external") / rel
    if not path.exists():
        with urlopen(url, timeout=90) as stream:
            raw = stream.read()
        assert hashlib.sha256(raw).hexdigest() == sha
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    assert digest(path) == sha
    return path


def language(split, count, excluded):
    grouped = defaultdict(list)
    with ZipFile(fetch("winogrande/source.zip", WIN_URL, WIN_SHA)) as z:
        for v in map(json.loads, z.open("winogrande_1.1/" + split + ".jsonl")):
            grouped[v["qID"].rsplit("-", 1)[0]].append(v)
    pairs = []
    for ident in sorted(grouped, key=lambda k: hashed("binding-491-" + k)):
        vs = grouped[ident]
        if len(vs) != 2 or {v["answer"] for v in vs} != {"1", "2"} or ident in excluded:
            continue
        options = [vs[0]["option1"], vs[0]["option2"]]
        if any(
            [v["option1"], v["option2"]] != options or v["sentence"].count("_") != 1
            for v in vs
        ):
            continue
        offered = options[::-1] if int(hashed(ident), 16) % 2 else options
        rows = []
        for v in vs:
            text = (
                v["sentence"]
                + '\n\nWhich phrase belongs in the blank, "'
                + offered[0]
                + '" or "'
                + offered[1]
                + '"?'
            )
            rows.append(
                {
                    "id": v["qID"],
                    "cluster": ident,
                    "family": "meaning",
                    "questions": [text],
                    "options": offered,
                    "gold": options[int(v["answer"]) - 1],
                }
            )
        pairs.append(rows)
        if len(pairs) == count:
            break
    assert len(pairs) == count
    return pairs


def previous_tables():
    names = (
        "tool-table-queries-01",
        "tool-sql-learning-panel",
        "table-parser-panel",
        "table-parser-training",
        "release-mixed-field-01",
    )
    return {
        k
        for n in names
        for k in re.findall(
            r'"table_id":\s*"([^"]+)"', (DATA / (n + ".json")).read_text()
        )
    }


def tables(split, count, excluded):
    path = fetch("wikisql/source.tar.bz2", SQL_URL, SQL_SHA)
    with tarfile.open(path, "r:bz2") as z:
        tables = {
            t["id"]: t
            for t in map(json.loads, z.extractfile(f"data/{split}.tables.jsonl"))
        }
        qs = defaultdict(list)
        for idx, q in enumerate(map(json.loads, z.extractfile(f"data/{split}.jsonl"))):
            sql = q["sql"]
            if sql["agg"] == 0 and len(sql["conds"]) == 1 and sql["conds"][0][1] == 0:
                qs[q["table_id"]].append((idx, q))
    pairs = []
    documents = {}
    for name in sorted(qs, key=lambda k: hashed("bind-table-971" + k)):
        if name in excluded:
            continue
        t = tables[name]
        cols = t["header"]
        if (
            len(qs[name]) < 2
            or not 2 <= len(cols) <= 14
            or len(cols) != len(set(cols))
            or not all(c.strip() for c in cols)
        ):
            continue
        if not 3 <= len(t["rows"]) <= 100 or any(
            len(r) != len(cols) for r in t["rows"]
        ):
            continue
        possibilities = []
        for idx, q in qs[name]:
            sql = q["sql"]
            j, _, literal = sql["conds"][0]
            s = sql["sel"]
            if s == j:
                continue
            hits = [
                r
                for r in t["rows"]
                if str(r[j]).strip().casefold() == str(literal).strip().casefold()
            ]
            if len(hits) != 1:
                continue
            key, gold = str(hits[0][j]), str(hits[0][s])
            if (
                not key
                or not gold
                or len(gold) > 100
                or key not in q["question"]
                or gold.casefold() == key.casefold()
            ):
                continue
            doc = "table-" + name + ".csv"
            possibilities.append(
                {
                    "id": f"wiki-{split}-{idx}",
                    "cluster": name,
                    "family": "table",
                    "document": doc,
                    "questions": [q["question"]],
                    "gold": gold,
                    "arguments": {
                        "document": doc,
                        "key_column": cols[j],
                        "key": key,
                        "value_column": cols[s],
                    },
                }
            )

        def distinct(ab):
            a, b = ab
            return (
                a["gold"].casefold() != b["gold"].casefold()
                and a["questions"] != b["questions"]
                and (a["arguments"]["key_column"], a["arguments"]["value_column"])
                != (b["arguments"]["key_column"], b["arguments"]["value_column"])
            )

        choices = filter(distinct, itertools.combinations(possibilities, 2))
        pair = min(
            choices,
            key=lambda ab: hashed("bind-pair" + ab[0]["id"] + ab[1]["id"]),
            default=None,
        )
        if pair is None:
            continue
        f = io.StringIO(newline="")
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(t["rows"])
        documents[pair[0]["document"]] = f.getvalue()
        pairs.append(pair)
        if len(pairs) == count:
            break
    if len(pairs) != count:
        raise ValueError(f"Only {len(pairs)} eligible {split} table pairs")
    return pairs, documents


def new(path, body):
    if path.exists():
        assert read(path) == body
    else:
        save(path, body)


def material():
    lang_test = language("dev", 24, set())
    test, docs = tables("test", 16, previous_tables())
    lang_train = language("train_xl", 384, {p[0]["cluster"] for p in lang_test})
    train, train_docs = tables(
        "train", 192, previous_tables() | {p[0]["cluster"] for p in test}
    )
    panel = [r for p in lang_test + test for r in p]
    panel.sort(key=lambda r: hashed("binding-order-740" + r["id"]))

    def grams(sentence):
        w = re.findall(r"\w+", sentence.casefold())
        return {tuple(w[i : i + 8]) for i in range(len(w) - 7)}

    held = {s for r in panel for s in grams(r["questions"][0].split("\n\n")[0])}
    held_text = {r["questions"][0].split("\n\n")[0].strip().casefold() for r in panel}
    tok = load_tokenizer(local_snapshot())
    with tempfile.TemporaryDirectory() as work:
        temp = Path(work)
        for file, content in {**docs, **train_docs}.items():
            (temp / file).write_bytes(content.encode())
        examples = []

        def source(r):
            tools_ = (
                tools([temp / r["document"]], keyed=True)
                if r["family"] == "table"
                else []
            )
            return (
                tools_,
                qwen_tools.declarations(tools_),
                [{"role": "user", "content": r["questions"][0]}],
            )

        for pair in lang_train + train:
            if any(
                r["questions"][0].split("\n\n")[0].strip().casefold() in held_text
                or grams(r["questions"][0].split("\n\n")[0]) & held
                for r in pair
            ):
                continue
            for i, r in enumerate(pair):
                funcs, defs, history = source(r)
                target = {"role": "assistant", "content": r["gold"]}
                prefix = {"id": r["id"], "tools": defs, "messages": history}
                if funcs:
                    action = qwen_tools.assistant_message(
                        [{"name": "csv_lookup", "arguments": r["arguments"]}]
                    )
                    observation = funcs[0].callback(**r["arguments"])
                    assert observation == {"value": r["gold"]}
                    examples.append(
                        {
                            **prefix,
                            "id": r["id"] + "-return",
                            "messages": [
                                *history,
                                action,
                                {"role": "tool", "content": compact(observation)},
                            ],
                            "target": target,
                        }
                    )
                    target = action
                examples.append(
                    {
                        **prefix,
                        "pair": r["cluster"],
                        "pair_order": i,
                        "thinking": "",
                        "target": target,
                    }
                )
        rehearsal = read(DATA / "native-function-training.json")["examples"]
        examples.extend(
            sorted(rehearsal, key=lambda r: hashed("bind-rehearse" + r["id"]))[:160]
        )
        for r in panel:
            funcs, defs, history = source(r)
            if funcs:
                assert funcs[0].callback(**r["arguments"]) == {"value": r["gold"]}
            r["initial_ids"] = tok.encode(
                qwen_tools.render(tok, history, defs), add_special_tokens=False
            )
    provenance = {
        "winogrande": {"url": WIN_URL, "sha256": WIN_SHA},
        "wikisql": {"url": SQL_URL, "sha256": SQL_SHA},
    }
    new(TRAIN, {"sources": provenance, "examples": examples})
    new(PANEL, {"sources": provenance, "documents": docs, "rows": panel})
    print(len(examples), "training turns", len(panel), "evaluation questions")


if __name__ == "__main__":
    material()
