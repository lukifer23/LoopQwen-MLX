"""Verify sealed evidence and build the reader-facing PDF and supplements."""

import argparse
import subprocess
import sys
from pathlib import Path

import typst
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def main(verify):
    from loopqwen.io import read

    subprocess.run([sys.executable, "scripts/paper_figures.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/paper_overlap.py"], cwd=ROOT, check=True)
    if verify:
        tracked = read(ROOT / "paper/figure-sources.json")
        tracked.update(
            {
                k: True
                for k, v in read(ROOT / "research/registry.json")["experiments"].items()
                if k.startswith(
                    (
                        "bridge-reading-",
                        "narrative-full-book-",
                        "transport-lossless-",
                        "raw-distillation-",
                        "cache-language-",
                        "tool-",
                        "schema-calls-",
                        "operand-planner-",
                        "release-mixed-field-",
                        "contrast-boolq-",
                        "register-rebalanced-",
                        "register-allocation-",
                        "native-language-release-02",
                    )
                )
                and v["archive"]
            }
        )
        for name in sorted(tracked):
            subprocess.run(
                [sys.executable, "scripts/research_archive.py", name, "--verify"],
                cwd=ROOT,
                check=True,
            )
    path = ROOT / "paper/loopqwen.pdf"
    typst.compile(str(ROOT / "paper/loopqwen.typ"), output=str(path), timestamp=0)
    pdf = PdfReader(path)
    text = [page.extract_text() for page in pdf.pages]
    if (
        len(text) < 6
        or any(len(t) < 200 for t in text)
        or not all(
            term in " ".join(text)
            for term in (
                "131,083",
                "124",
                "Executable tool use",
                "Limitations",
                "Sources and artifact guide",
            )
        )
    ):
        raise ValueError("Missing PDF content")
    print(f"Verified {len(text)} text-bearing PDF pages in {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Independently recount the archives supporting the figures",
    )
    main(parser.parse_args().verify)
