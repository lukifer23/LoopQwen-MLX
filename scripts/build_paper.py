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
    subprocess.run([sys.executable, "scripts/folded_overlap.py"], cwd=ROOT, check=True)
    if verify:
        subprocess.run(
            [sys.executable, "scripts/report_distribution_bytes.py"],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [sys.executable, "scripts/research_registry.py", "--verify-files"],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [sys.executable, "scripts/selective_source_audit.py", "--verify"],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [sys.executable, "scripts/verify_user_sessions.py"], cwd=ROOT, check=True
        )
        registry = read(ROOT / "research/registry.json")
        for name, study in sorted(registry["experiments"].items()):
            if study["archive"]:
                subprocess.run(
                    [sys.executable, "scripts/research_archive.py", name, "--verify"],
                    cwd=ROOT,
                    check=True,
                )
        for location in sorted(registry["session_results"]):
            session, name = location.split("/", 1)
            subprocess.run(
                [
                    sys.executable,
                    "scripts/research_archive.py",
                    name,
                    "--recovered-session",
                    session,
                    "--verify",
                ],
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
        help="Verify every published archive and reconstruct the installed examples",
    )
    main(parser.parse_args().verify)
