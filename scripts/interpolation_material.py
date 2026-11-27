"""Freeze original opposite-answer Winograd pairs and real package release inventories."""

import csv
import hashlib
import io
import itertools
import subprocess
import tempfile
import tomllib
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen

from mlx_lm.utils import load_tokenizer

from loopqwen import qwen_tools
from loopqwen.calculator_tool import tool
from loopqwen.csv_tools import tools
from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot

URL = "https://cs.nyu.edu/~davise/papers/WinogradSchemas/WSCollection.xml"
XML_SHA = "98c845a5b33c109f2c49742e726fac592bbb1fdc3b3706b604de12f9c9f58c91"
REV = "d58c3fa"
PATH = Path("research/data/instruction-interpolation-panel.json")


def hashed(string):
    return hashlib.sha256(string.encode()).hexdigest()


def text(node):
    return " ".join("".join(node.itertext()).split())


def sentences():
    path = Path("data/research_external/winograd/WSCollection.xml")
    if not path.exists():
        with urlopen(URL, timeout=30) as response:
            raw = response.read()
        assert hashlib.sha256(raw).hexdigest() == XML_SHA
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(raw)
    assert digest(path) == XML_SHA
    nodes = ET.parse(path).getroot().findall("schema")
    items = []
    for i, n in enumerate(nodes):
        options = [text(x) for x in n.findall("answers/answer")]
        letter = text(n.find("correctAnswer")).strip(". ")
        items.append(
            (
                i,
                text(n.find("text")),
                text(n.find("text/pron")),
                text(n.find("quote")),
                options,
                letter,
                text(n.find("source")),
            )
        )
    pairs = []
    used = set()
    for a, b in itertools.pairwise(items):
        if (
            a[0] in used
            or b[0] in used
            or a[4] != b[4]
            or len(a[4]) != 2
            or a[2] != b[2]
            or a[6] != b[6]
            or {a[5], b[5]} != {"A", "B"}
        ):
            continue
        if (
            a[1] == b[1]
            or SequenceMatcher(None, a[1], b[1], autojunk=False).ratio() < 0.80
        ):
            continue
        pairs.append((a, b))
        used.update((a[0], b[0]))
    rows = []
    for a, b in sorted(pairs, key=lambda p: hashed("wsc-two-actions-01" + p[0][1]))[
        :20
    ]:
        options = list(a[4])
        cluster = "wsc-" + str(a[0])
        if int(hashed(cluster), 16) % 2:
            options.reverse()
        for idx, s, pron, quote, choices, letter, author in (a, b):
            question = f'{s}\n\nIn the phrase "{quote}", who or what does "{pron}" refer to? Choose between "{options[0]}" and "{options[1]}". Give the name or phrase.'
            rows.append(
                {
                    "id": "wsc-" + str(idx),
                    "cluster": cluster,
                    "family": "reference",
                    "questions": [question],
                    "options": options,
                    "gold": choices["AB".index(letter)],
                    "author": author,
                }
            )
    assert len(rows) == 40
    return rows


def inventories():
    raw = subprocess.check_output(["git", "show", REV + ":uv.lock"])
    earlier = {
        w["cluster"]
        for w in read("research/data/report-action-panel.json")["rows"]
        if w["family"] == "distribution"
    }
    packages = [
        p
        for p in tomllib.loads(raw.decode())["package"]
        if p["name"] not in earlier and len(p.get("wheels", [])) >= 2
    ]
    docs = {}
    rows = []
    for p in sorted(
        packages, key=lambda v: hashed("keyed-release-inventory-316" + v["name"])
    ):
        name = p["name"]
        file = name + ".csv"
        buf = io.StringIO(newline="")
        footer = csv.writer(buf)
        footer.writerow(["filename", "bytes", "sha256"])
        wheels = []
        for w in p["wheels"]:
            nm = unquote(urlsplit(w["url"]).path.rsplit("/", 1)[1])
            footer.writerow([nm, w["size"], w["hash"].removeprefix("sha256:")])
            wheels.append((nm, w["size"]))
        w1, w2 = sorted(wheels, key=lambda w: hashed("resolution-642" + w[0]))[:2]
        small, large = sorted([w1, w2], key=lambda w: w[1])
        if small[1] == large[1]:
            continue
        docs[file] = buf.getvalue()
        qs = [
            f"Please look up the size in bytes of {small[0]} in {file}.",
            f"For the same release, how many more bytes does {large[0]} take than the file we just checked? Look up its size and calculate the difference.",
        ]
        rows.append(
            {
                "id": name,
                "cluster": name,
                "family": "inventory",
                "questions": qs,
                "document": file,
                "keys": [small[0], large[0]],
                "values": [small[1], large[1]],
                "gold": [str(small[1]), str(large[1] - small[1])],
            }
        )
        if len(rows) == 8:
            break
    assert len(rows) == 8
    return rows, docs, hashlib.sha256(raw).hexdigest()


def material():
    rows = sentences()
    more, docs, lock = inventories()
    rows.extend(more)
    rows.sort(key=lambda w: hashed("common-order-704" + w["id"]))
    tok = load_tokenizer(local_snapshot())
    with tempfile.TemporaryDirectory() as tmp:
        for name, body in docs.items():
            (Path(tmp) / name).write_bytes(body.encode())
        for w in rows:
            funcs = (
                tools([Path(tmp) / w["document"]], keyed=True) + [tool()]
                if w["family"] == "inventory"
                else []
            )
            w["initial_ids"] = tok.encode(
                qwen_tools.render(
                    tok,
                    [{"role": "user", "content": w["questions"][0]}],
                    qwen_tools.declarations(funcs),
                ),
                add_special_tokens=False,
            )
    d = {
        "rows": rows,
        "documents": docs,
        "source": URL,
        "source_sha256": XML_SHA,
        "lock_revision": REV,
        "lock_sha256": lock,
    }
    if PATH.exists():
        assert read(PATH) == d
    else:
        save(PATH, d)
    return d


if __name__ == "__main__":
    d = material()
    print(len(d["rows"]), "sessions", list(d["documents"]))
