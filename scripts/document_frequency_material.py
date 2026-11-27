"""Freeze long, complete Git and uv documents and source-located training examples."""

import argparse
import hashlib
import os
import shutil
import subprocess
import tomllib
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit
from urllib.request import urlopen
from zipfile import ZipFile

from mlx_lm.utils import load_tokenizer

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.tool_session import prompt

ROOT = Path("research/data")
REVISIONS = (
    "5bd60c2c678b3f527805e8050f5104a7bc9c78e0",
    "4c4ecf26a04afd71fcccc702b2a1ec20fa8519e7",
)
SQUAD = ROOT / "passage-copy-train.json"
ZIP = Path("data/research_external/source-balanced/cpython.zip")


CPYTHON = "4a5632fbf9bf59477c540e3f53fa7cdbeea3e3f5"
ZIP_SHA = "a9a7cb0389b99be7076f473f5d19e3b835276eb63f1d6294426d3ec93238b5e3"


def check_source():
    if not ZIP.exists():
        ZIP.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=ZIP.parent, delete_on_close=False) as tmp:
            with urlopen(
                "https://codeload.github.com/python/cpython/zip/" + CPYTHON, timeout=90
            ) as response:
                shutil.copyfileobj(response, tmp)
            tmp.close()
            target = Path(tmp.name)
            if digest(target) != ZIP_SHA:
                raise ValueError("CPython source archive checksum differs")
            # Publish a verified copy atomically; the temporary file is owned by its context.
            try:
                os.link(target, ZIP)
            except FileExistsError:
                pass
    if digest(ZIP) != ZIP_SHA:
        raise ValueError("CPython source archive checksum differs")


def key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args]).decode()


def documents():
    revs = [git("rev-parse", r).strip() for r in REVISIONS]
    trees = [git("ls-tree", "-r", "-l", r) for r in revs]
    sizes = [
        {
            line.split("\t")[1]: line.split("\t")[0].split()[-1]
            for line in s.splitlines()
        }
        for s in trees
    ]
    changed = [
        p
        for p in sizes[0].keys() & sizes[1].keys()
        if p.startswith(("paper/", "docs/"))
        and p.endswith((".md", ".typ", ".csv"))
        and min(int(v[p]) for v in sizes) > 1000
        and sizes[0][p] != sizes[1][p]
    ]
    name = min(changed, key=lambda p: key("whole-document-field-2376" + p))
    for r, s, table in zip(revs, trees, sizes, strict=True):
        gold = table[name]
        assert git("cat-file", "-s", r + ":" + name).strip() == gold
        line = next(v for v in s.splitlines() if v.endswith("\t" + name))
        lo = s.index(line) + line.rindex(gold + "\t")
        yield (
            r,
            "tree",
            s,
            "How many bytes does the listing give for "
            + name
            + "? Give the byte count.",
            gold,
            lo,
        )
    s = git("show", revs[1] + ":uv.lock")
    wheels = [
        w
        for p in tomllib.loads(s)["package"]
        for w in p.get("wheels", [])
        if w["url"].endswith("py3-none-any.whl")
    ]
    w = min(wheels, key=lambda x: key("whole-document-lock-2377" + x["url"]))
    filename = Path(urlsplit(w["url"]).path).name
    gold = str(w["size"])
    line = next(v for v in s.splitlines() if w["url"] in v)
    lo = s.index(line) + line.index("size = " + gold) + len("size = ")
    yield (
        revs[1] + "-uv",
        "lockfile",
        s,
        "According to this lockfile, what is the size in bytes of "
        + filename
        + "? Give the byte count.",
        gold,
        lo,
    )


def panel():
    tokenizer = load_tokenizer(local_snapshot())
    rows = []
    for ident, family, source, question, gold, lo in documents():
        user = source + "\n\n" + question
        rendered = prompt(tokenizer, [{"role": "user", "content": user}])
        start = rendered.index(source) + lo
        encoded = tokenizer._tokenizer(
            rendered, add_special_tokens=False, return_offsets_mapping=True
        )
        ids = encoded["input_ids"]
        span = [
            i
            for i, (a, b) in enumerate(encoded["offset_mapping"])
            if b > start and a < start + len(gold)
        ]
        assert tokenizer.decode(ids[span[0] : span[-1] + 1]).strip() == gold
        assert len(ids) >= (131072 if family == "tree" else 32768)
        rows.append(
            {
                "id": ident,
                "family": family,
                "question": question,
                "gold": gold,
                "source_sha256": key(source),
                "gold_start": span[0],
                "gold_stop": span[-1] + 1,
                "sha256": fingerprint(ids),
                "prompt_ids": ids,
            }
        )
    return {"panel": rows}


def training():
    check_source()
    human = {
        w["id"]: w
        for w in read(SQUAD)["rows"]
        if w["domain"] == "SQuAD" and not w["unanswerable"]
    }
    ids = sorted(human, key=lambda i: key("span-doc-training-2382" + i))
    rows = []
    for i, ident in enumerate(ids[:384]):
        # Other paragraphs and random answer position prevent an absolute location rule.
        group = [ident] + [ids[(i + off * 47) % len(ids)] for off in range(1, 12)]
        group.sort(key=lambda x: key(ident + x))
        rows.append({"id": ident, "kind": "human", "paragraphs": group})
    with ZipFile(ZIP) as z:
        files = [x for x in z.infolist() if not x.is_dir() and x.file_size > 0]
        names = [x.filename.split("/", 1)[1] for x in files]
        for i in sorted(
            range(len(files)), key=lambda i: key("span-doc-training-2383" + names[i])
        )[:256]:
            a = max(0, i - int(key(names[i])[:4], 16) % 40)
            indices = list(range(a, min(a + 40, len(files))))
            rows.append(
                {
                    "id": names[i],
                    "kind": "metadata",
                    "records": [[files[j].file_size, names[j]] for j in indices],
                }
            )
    return {"rows": rows, "sources": {str(SQUAD): digest(SQUAD), str(ZIP): digest(ZIP)}}


def training_documents(frozen):
    human = {w["id"]: w for w in read(SQUAD)["rows"]}
    for r in frozen["rows"]:
        if r["kind"] == "human":
            w = human[r["id"]]
            ps = [human[x]["passage"] for x in r["paragraphs"]]
            idx = r["paragraphs"].index(r["id"])
            start = sum(len(p) + 1 for p in ps[:idx]) + w["answer_char"]
            yield "\n".join(ps), w["question"], start, start + len(w["answer"])
        else:
            lines = [f"{n}\t{name}" for n, name in r["records"]]
            idx = next(i for i, (_, name) in enumerate(r["records"]) if name == r["id"])
            start = sum(len(p) + 1 for p in lines[:idx])
            yield (
                "\n".join(lines),
                f"What is the uncompressed size of {r['id']}?",
                start,
                start + len(str(r["records"][idx][0])),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-request",
        type=int,
        choices=(1, 2, 3),
        help="Print one exact user message for the installed generator",
    )
    args = parser.parse_args()
    if args.show_request:
        _, _, source, question, _, _ = list(documents())[args.show_request - 1]
        print(source + "\n\n" + question, end="")
        raise SystemExit(0)
    for name, data in [("panel", panel()), ("training", training())]:
        path = ROOT / f"document-frequency-{name}.json"
        if path.exists():
            assert read(path) == data
        else:
            save(path, data)
    for w in read(ROOT / "document-frequency-panel.json")["panel"]:
        print(w["family"], len(w["prompt_ids"]), w["question"], w["gold"])
