"""Typeset, join and verify the later research editions without MLX or a source checkout."""

import argparse
import hashlib
import io
from pathlib import Path

import typst
from pypdf import PdfReader, PdfWriter

EDITIONS = {
    "revision-evidence": (
        "loopqwen-complete.pdf",
        "72e493ab69b2f6bd352327c7db2a1ce3fa23ef7b1eab29f4ee529e0d16a6d3f7",
        "loopqwen-research.pdf",
        "LoopQwen-MLX: recurrent depth, edited evidence and executable tools",
        "Current study · Did the answer change with the article?",
        "Field update and original technical paper",
    ),
    "file-evidence": (
        "loopqwen-research.pdf",
        "fde9a38e1998228d3b907265e5fed96a8f8e8b59dd428c6c2734315a5a132c1a",
        "loopqwen-report.pdf",
        "LoopQwen-MLX: file evidence, reasoning, tools and full-context inference",
        "Current study · When only the file changes",
        "Recurrent reasoning, field studies and technical paper",
    ),
}


def edition(folder, name, *, verify=False):
    base, sha, dest, title, heading, history = EDITIONS[name]
    original = (folder / base).read_bytes()
    assert hashlib.sha256(original).hexdigest() == sha, base
    current = typst.compile(str(folder / (name + ".typ")), timestamp=0)
    supplement = folder / (name + ".pdf")
    if verify:
        assert supplement.read_bytes() == current, name
    else:
        supplement.write_bytes(current)
    writer = PdfWriter()
    writer.add_metadata({"/Title": title, "/Author": "LoopQwen-MLX project"})
    pages = []
    for src, bookmark in ((current, heading), (original, history)):
        doc = PdfReader(io.BytesIO(src))
        texts = [p.extract_text() for p in doc.pages]
        assert all(len(t) > 200 for t in texts)
        pages.extend(texts)
        writer.append(doc, outline_item=bookmark)
    stream = io.BytesIO()
    writer.write(stream)
    raw = stream.getvalue()
    assert [p.extract_text() for p in PdfReader(io.BytesIO(raw)).pages] == pages
    path = folder / dest
    if verify:
        assert path.read_bytes() == raw, dest
    else:
        path.write_bytes(raw)
    print(len(pages), "pages", hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    for name in EDITIONS:
        edition(args.folder, name, verify=args.verify)
