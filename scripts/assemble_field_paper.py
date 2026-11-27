"""Typeset the two field supplements and combine them with the original 64-page publication."""

import argparse
import hashlib
import io
from pathlib import Path

import typst
from pypdf import PdfReader, PdfWriter

ORIGINAL = "87ce7d863edd5750c9a99ca02cf1e202a7f7778e89ec07bb82b3095588c9e2ad"


def assemble(folder, verify=False):
    original = (folder / "loopqwen.pdf").read_bytes()
    assert hashlib.sha256(original).hexdigest() == ORIGINAL
    writer = PdfWriter()
    writer.add_metadata(
        {
            "/Title": "Where shared decoder depth earns its cost – complete research report",
            "/Author": "LoopQwen-MLX project",
        }
    )
    parts = []
    for name, bookmark in [
        ("role-binding", "Current study · Learning to bind the right argument"),
        (
            "instruction-interpolation",
            "Prior field comparison · When the referent changes",
        ),
        (
            "loopqwen",
            "Main technical report · Where shared decoder depth earns its cost",
        ),
    ]:
        raw = (
            original
            if name == "loopqwen"
            else typst.compile(str(folder / (name + ".typ")), timestamp=0)
        )
        path = folder / (name + ".pdf")
        if verify:
            assert raw == path.read_bytes(), name
        elif name != "loopqwen":
            path.write_bytes(raw)
        data = PdfReader(io.BytesIO(raw))
        texts = [page.extract_text() for page in data.pages]
        assert all(len(t) > 200 for t in texts)
        writer.append(data, outline_item=bookmark)
        parts.extend(texts)
    out = io.BytesIO()
    writer.write(out)
    raw = out.getvalue()
    assert [p.extract_text() for p in PdfReader(io.BytesIO(raw)).pages] == parts
    path = folder / "loopqwen-complete.pdf"
    if verify:
        assert raw == path.read_bytes(), "combined paper"
    else:
        path.write_bytes(raw)
    print(len(parts), "pages", hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    assemble(args.folder, args.verify)
