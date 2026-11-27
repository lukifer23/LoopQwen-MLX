"""Verify sealed evidence and build the reader-facing PDF and supplements."""

import argparse
import subprocess
import sys
from pathlib import Path

import typst
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def main(verify):
    import hashlib

    from assemble_field_paper import ORIGINAL

    from loopqwen.io import read

    current = (ROOT / "research/results/role-binding-01/decision.json").is_file()
    if current:
        subprocess.run(
            [sys.executable, "scripts/paper_induction_evidence.py", "--verify"],
            cwd=ROOT,
            check=True,
        )
    else:
        subprocess.run(
            [sys.executable, "scripts/paper_induction_evidence.py"],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [sys.executable, "scripts/paper_figures.py"], cwd=ROOT, check=True
        )
        subprocess.run(
            [sys.executable, "scripts/paper_overlap.py"], cwd=ROOT, check=True
        )
        subprocess.run(
            [sys.executable, "scripts/folded_overlap.py"], cwd=ROOT, check=True
        )
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
    historical = typst.compile(str(ROOT / "paper/loopqwen.typ"), timestamp=0)
    if current:
        if (
            hashlib.sha256(historical).hexdigest() != ORIGINAL
            or path.read_bytes() != historical
        ):
            raise ValueError(
                "Historical manuscript no longer reproduces the original field kit"
            )
    else:
        path.write_bytes(historical)
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
    print(f"Verified {len(text)} text-bearing PDF pages in {path}", flush=True)
    if current:
        for args in (
            ["scripts/binding_source_check.py"],
            ["scripts/report_distribution_bytes.py", "--keyed-inventories"],
            ["scripts/role_binding_exposure.py"],
            ["scripts/role_binding_reading.py"],
            ["scripts/paper_instruction_interpolation.py"],
            ["scripts/paper_role_binding.py"],
        ):
            subprocess.run([sys.executable, *args], cwd=ROOT, check=True)
    if (ROOT / "research/results/revision-evidence-01/decision.json").is_file():
        for module in (
            "scripts.revision_source_check",
            "scripts.revision_answer_reading",
            "scripts.paper_revision_evidence",
        ):
            subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True)

    field = ROOT / "research/results/file-evidence-accumulated-01/decision.json"
    if field.is_file() and read(field).get("status") != "interrupted":
        for args in (
            ["scripts.revision_source_check", "--file-evidence"],
            ["scripts.file_evidence_audit", "--sources"],
            ["scripts.file_evidence_checkpoints"],
            ["scripts.file_evidence_exposure"],
            ["scripts.file_evidence_overlap"],
            ["scripts.file_evidence_audit"],
            ["scripts.paper_file_evidence"],
        ):
            subprocess.run([sys.executable, "-m", *args], cwd=ROOT, check=True)

    cross = ROOT / "research/results/cross-view-reading-01/decision.json"
    if cross.is_file() and read(cross).get("status") != "interrupted":
        subprocess.run(
            [sys.executable, "-m", "scripts.file_evidence_checkpoints", "--cross-view"],
            cwd=ROOT,
            check=True,
        )
        for command in (
            "cross_view_source_check",
            "cross_view_exposure",
            "cross_view_reading",
            "paper_cross_view",
        ):
            subprocess.run(
                [sys.executable, "-m", "scripts." + command], cwd=ROOT, check=True
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify every published archive and reconstruct the installed examples",
    )
    main(parser.parse_args().verify)
