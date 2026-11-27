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
    "cross-view": (
        "loopqwen-report.pdf",
        "7b48e85701c2566af36469851ac5daffcd726186ac45e19e51f24122a0a70e69",
        "loopqwen-study.pdf",
        "LoopQwen-MLX: cross-view reading, real tools and full-context inference",
        "Current study · Can the loop read its own tool returns?",
        "Changed files, independent reasoning and technical paper",
    ),
    "reasoned-evidence": (
        "loopqwen-study.pdf",
        "c4e8346d252e46f39731bc1849686372b1aafa73aee4b90c724d3f1bae092142",
        "loopqwen-findings.pdf",
        "LoopQwen-MLX: qualifying reasoning supervision for recurrent tool use",
        "Current study · Qualifying explanations before recurrent training",
        "Cross-view reading, tools, full context and earlier technical evidence",
    ),
    "proof-curriculum": (
        "loopqwen-findings.pdf",
        "1622e82a046cd1da9270b22159a8198d03600c3eb485d151cc6721681707c9ef",
        "loopqwen-proof-study.pdf",
        "LoopQwen-MLX: second-visit learning, human reasoning, tool calls and full context",
        "Current study · Training the repeated visit to improve a verdict",
        "Explanation supervision, cross-view reading and full technical report",
    ),
    "field-selection": (
        "loopqwen-proof-study.pdf",
        "ccb0bb7a5fbcb9e1677722380a0a4a6e035768ff873cccf496573b1d9cce01f3",
        "loopqwen-field-study.pdf",
        "LoopQwen-MLX: recurrent field selection, human judgments and actual compatibility decisions",
        "Current study · Choosing and interpreting the right field",
        "Previous reasoning studies and the complete technical report",
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
        if (
            name
            in {
                "cross-view",
                "reasoned-evidence",
                "proof-curriculum",
                "field-selection",
            }
            and not (args.folder / (name + "-paper.json")).is_file()
        ):
            continue
        edition(args.folder, name, verify=args.verify)
