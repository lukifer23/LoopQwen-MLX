"""Freeze source-linked specialist questions and run the direct standalone comparison."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from loopqwen.io import read, save, digest
from loopqwen.identity import fingerprint
from loopqwen.research.cross_view_study import tokenizer
from scripts.middle_depth_budget import BASE, ROOT
from scripts.middle_depth_development import prompt_ids
from scripts.middle_depth_local_workflows import collect, frozen, public_test_file
from scripts.middle_depth_report_tools import reports
from scripts.middle_depth_wave import dispatch
from scripts.program_reasoning_material import source

WORK = Path("outputs/middle-depth/specialist-transfer")
PLAN = Path("research/diagnostics/middle-depth/specialist-transfer.md")
QMSUM = "83d7768c1f2b4dfeb091385d3dc7e239b8e5bb7e"
MEETINGS = {
    "Bed003": "959b0a710628f786f64c35f1462717f234b03ba34f2e170d3bb13be72f3321f8",
    "ES2004a": "911e56db0d8482f1e730bae0a8fcf0ab0095c94abb000ba0516e975fa6e48b31",
}
SUMMARY = "Compare the early and terminal reports for the loop at the 128-generated-token setting. Write a short handoff explaining which family improved, which declined, which stayed flat, and what happened to the total. Include counts so I can check."
CSV = "Give me just the family results as CSV, with the header family,early_correct,terminal_correct and one row per family."
PROGRAM = """import glob, json
for name in sorted(glob.glob('*.json')):
    for row in json.load(open(name)):
        if row['arm'] == 'loop' and row['cap'] == 128:
            print(name, row['family'], row['success'], row['episodes'])
"""


def providers(kind, folder, stack):
    return reports(folder / "reports") if kind == "files" else []


def build(folder):
    cases = []
    report_dir = folder / "reports"
    report_dir.mkdir()
    counts = {}
    for stage in ("early", "terminal"):
        src = ROOT / "review" / stage / "summary.json"
        shutil.copyfile(src, report_dir / (stage + ".json"))
        counts[stage] = {
            r["family"]: r["success"]
            for r in read(src)
            if r["arm"] == "loop" and r["cap"] == 128
        }
    terminal, python, _ = reports(report_dir)
    listing = terminal.callback("ls")
    observed = python.callback(PROGRAM)
    assert listing["stdout"].splitlines() == ["early.json", "terminal.json"]
    assert observed["exit_code"] == 0
    expected = {
        (stage + ".json", family, str(n), "9")
        for stage, rows in counts.items()
        for family, n in rows.items()
    }
    assert {tuple(line.split()) for line in observed["stdout"].splitlines()} == expected
    save(
        folder / "witness.json",
        {"terminal": listing, "program": PROGRAM, "python": observed},
    )
    ref = json.dumps(counts) + ". Overall supported answers remained 5 of 27."
    for i, (kind, introduction) in enumerate(
        [
            ("files", "Use Python to read the two local reports. "),
            (
                "supplied",
                "Here are the rows I obtained using Python (filename, family, supported, evaluated).\n"
                + observed["stdout"]
                + "\n",
            ),
        ]
    ):
        cases.append(
            dict(
                id=kind,
                kind=kind,
                block=i,
                prompts=[introduction + SUMMARY, CSV],
                refs=[ref, ref],
            )
        )

    corpus = Path("data/research_external/qmsum")
    corpus.mkdir(parents=True, exist_ok=True)
    for i, (name, h) in enumerate(MEETINGS.items()):
        url = f"https://raw.githubusercontent.com/Yale-LILY/QMSum/{QMSUM}/data/ALL/test/{name}.json"
        src = source("qmsum/" + name + ".json", url, h)
        d = read(src)
        transcript = d["meeting_transcripts"]
        q = d["specific_query_list"][0]
        dialogue = "\n".join(f"{r['speaker']}: {r['content']}" for r in transcript)
        evidence = [transcript[int(a) : int(b) + 1] for a, b in q["relevant_text_span"]]
        assert all(evidence)
        cases.append(
            dict(
                id=name,
                kind="meeting",
                block=i,
                prompts=[dialogue + "\n\n" + q["query"]],
                refs=[q["answer"]],
                evidence=evidence,
                spans=q["relevant_text_span"],
                source=h,
                utterances=len(transcript),
            )
        )
    official = [json.loads(s) for s in public_test_file().read_text().splitlines()]
    changes = [
        (0, "16 eggs", "18 eggs", lambda n: (n - 3 - 4) * 2, 16, 18),
        (330, "$43", "$47", lambda n: n - 4 - 8, 43, 47),
        (660, "70", "68", lambda n: (n - 2 * 20) // 2, 70, 68),
        (
            990,
            "high for 4 hours",
            "high for 5 hours",
            lambda n: 3 * 2 + n * 3 - 2 * 2,
            4,
            5,
        ),
    ]
    for i, (index, old, new, operation, a, b) in enumerate(changes):
        q = official[index]
        assert (
            q["question"].count(old) == 1
            and str(operation(a)) == q["answer"].rsplit("####", 1)[-1].strip()
        )
        cases.append(
            dict(
                id=f"gsm-{index}",
                kind="math",
                block=i // 2,
                prompts=[q["question"].replace(old, new)],
                refs=[str(operation(b))],
                source_index=index,
                original_answer=operation(a),
            )
        )
    tok = tokenizer({"checkpoint": str(BASE)})
    for case in cases:
        with ExitStack() as stack:
            ids = prompt_ids(
                tok,
                [dict(role="user", content=case["prompts"][0])],
                providers(case["kind"], folder, stack),
            )
        case.update(opening=fingerprint(ids), input_positions=len(ids))
    files = {
        str(p.relative_to(folder)): digest(p) for p in folder.rglob("*") if p.is_file()
    }
    save(
        folder / "cohort.json",
        dict(
            plan=digest(PLAN),
            files=files,
            cases=cases,
            gsm=digest(public_test_file()),
            qmsum_revision=QMSUM,
        ),
    )


def prepare():
    if not WORK.exists():
        WORK.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=WORK.parent) as temp:
            d = Path(temp) / "work"
            d.mkdir()
            build(d)
            d.rename(WORK)
    cohort = frozen(WORK, PLAN)
    save(
        ROOT / "specialist-inputs.json",
        {
            "plan": digest(PLAN),
            "cohort": digest(WORK / "cohort.json"),
            "qmsum": {"revision": QMSUM, "sources": MEETINGS},
            "gsm_test": cohort["gsm"],
            "inputs": [
                {k: r[k] for k in ("id", "kind", "block", "opening", "input_positions")}
                for r in cohort["cases"]
            ],
        },
    )


def panel():
    prepare()
    for block, arms in enumerate((("base", "loop"), ("loop", "base"))):
        for arm in arms:
            dispatch("specialist", "collect", "--arm", arm, "--block", block)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "panel", "collect"))
    parser.add_argument("--arm", choices=("base", "loop"))
    parser.add_argument("--block", type=int, choices=(0, 1))
    a = parser.parse_args()
    if a.action == "prepare":
        prepare()
    elif a.action == "panel":
        panel()
    else:
        collect(
            a.arm,
            a.block,
            folder=WORK,
            plan=PLAN,
            catalog=providers,
            job="specialist-transfer",
        )
