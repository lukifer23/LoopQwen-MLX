"""Freeze and run project-local deployment jobs and a separate public arithmetic check."""

import argparse
import json
import os
import shutil
import sqlite3
from contextlib import ExitStack
from pathlib import Path
from time import monotonic

import mlx.core as mx

from loopqwen.calculator_tool import tool as calculator
from loopqwen.document_tools import documents
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.relational_tools import tools as relational_tools
from loopqwen.tool_cli import open_assistant
from scripts.middle_depth_budget import BASE, worker
from scripts.middle_depth_development import prompt_ids
from scripts.middle_depth_review import first_visible
from scripts.middle_depth_standalone import MODELS, check
from scripts.middle_depth_wave import dispatch

LOCAL = Path("outputs/middle-depth/local-workflows")
PLAN = Path("research/diagnostics/middle-depth/local-workflows.md")
GSM_SHA = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"


def public_test_file():
    from scripts.program_reasoning_material import GSM_REV, source

    archived = Path("data/phase2_external/gsm_test.jsonl")
    if archived.is_file():
        if digest(archived) != GSM_SHA:
            raise ValueError(
                "Archived GSM8K split differs from the official test bytes"
            )
        return archived
    url = f"https://raw.githubusercontent.com/openai/grade-school-math/{GSM_REV}/grade_school_math/data/test.jsonl"
    return source("gsm-test-official.jsonl", url, GSM_SHA)


def providers(kind, folder, stack):
    if kind == "resume":
        return [
            *stack.enter_context(
                documents([folder / "archived.json", folder / "matched.json"])
            ),
            calculator(),
        ]
    if kind == "transfer":
        return [*relational_tools([folder / "inventory.db"]), calculator()]
    return []


def prepare():
    from tempfile import TemporaryDirectory
    from loopqwen.research.cross_view_study import tokenizer

    if LOCAL.exists():
        return frozen()
    LOCAL.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=LOCAL.parent) as work:
        folder = Path(work) / "cohort"
        folder.mkdir()
        states = {}
        for name, run in (("archived", "4619"), ("matched", "matched-4619")):
            root = Path("checkpoints/middle-depth") / run / "loop"
            ptr = read(root / "latest.json")
            assert ptr["state"] == digest(root / ptr["path"] / "state.json")
            for key in ("path", "previous"):
                location = root / ptr[key]
                st = read(location / "state.json")
                assert all(digest(location / f) == h for f, h in st["hashes"].items())
            states[name] = [
                read(root / ptr[k] / "state.json")["step"] for k in ("path", "previous")
            ]
            shutil.copyfile(root / "latest.json", folder / f"{name}.json")
        sizes, backbone = {}, {}
        db = sqlite3.connect(folder / "inventory.db")
        try:
            db.execute(
                "CREATE TABLE files(model TEXT, filename TEXT, bytes INTEGER, sha256 TEXT, PRIMARY KEY(model,filename))"
            )
            for arm in ("base", "loop"):
                directory = MODELS / arm
                manifest = check(directory, arm)["files"]
                raw = [
                    (arm, f, (directory / f).stat().st_size, h)
                    for f, h in sorted(manifest.items())
                ]
                db.executemany("INSERT INTO files VALUES(?,?,?,?)", raw)
                sizes[arm] = sum(n for _, _, n, _ in raw)
                backbone[arm] = manifest["model.safetensors"]
            db.commit()
        finally:
            db.close()
        farther = max(states, key=lambda n: states[n][0])
        limit, previous = states[farther]
        larger = max(sizes, key=sizes.get)
        cases = [
            {
                "id": "resume",
                "block": 0,
                "kind": "resume",
                "prompts": [
                    "Read the archived and matched loop restart pointers. Which run reached the later optimizer step, and how many updates farther did it get?",
                    "For the farther-along run, which saved step is its fallback, and how many updates would I lose by rolling back?",
                ],
                "refs": [
                    f"{farther} reached {limit}, ahead by {limit - min(s[0] for s in states.values())} updates. Read both actual pointers.",
                    f"The saved fallback is step {previous} and rolling back costs {limit - previous} updates. Ground this in the actual pointer.",
                ],
            },
            {
                "id": "transfer",
                "block": 1,
                "kind": "transfer",
                "prompts": [
                    "I'm copying one of the standalone model folders to another Mac. Check the file inventory. Which folder needs more room, and how many extra bytes would I copy compared with the other?",
                    "Check the model.safetensors entries specifically. Do the two backbone files have the same SHA-256? Give the first 12 hex digits for each.",
                ],
                "refs": [
                    f"{larger} needs {max(sizes.values()) - min(sizes.values())} extra bytes. Use the file inventory.",
                    f"Backbone digest equality is {backbone['base'] == backbone['loop']}. Base prefix {backbone['base'][:12]}, loop prefix {backbone['loop'][:12]}. Verify against file entries.",
                ],
            },
        ]
        public = [
            json.loads(line) for line in public_test_file().read_text().splitlines()
        ]
        assert len(public) == 1319
        for j, index in enumerate((0, 330, 660, 990)):
            item = public[index]
            cases.append(
                {
                    "id": f"gsm-{index}",
                    "block": j // 2,
                    "kind": "math",
                    "prompts": [item["question"]],
                    "refs": [item["answer"].rsplit("#### ", 1)[1].strip()],
                }
            )
        tok = tokenizer({"checkpoint": str(BASE)})
        for case in cases:
            with ExitStack() as stack:
                tools = providers(case["kind"], folder, stack)
                ids = prompt_ids(
                    tok, [{"role": "user", "content": case["prompts"][0]}], tools
                )
                case["opening"] = fingerprint(ids)
        files = {f.name: digest(f) for f in folder.iterdir()}
        save(
            folder / "cohort.json",
            {"plan": digest(PLAN), "gsm": GSM_SHA, "files": files, "cases": cases},
        )
        folder.rename(LOCAL)
    return frozen()


def frozen():
    cohort = read(LOCAL / "cohort.json")
    if cohort["plan"] != digest(PLAN) or any(
        digest(LOCAL / f) != h for f, h in cohort["files"].items()
    ):
        raise ValueError("Local evaluation inputs changed")
    return cohort


def collect(arm, block):
    cohort = frozen()
    with worker("short", f"local-workflows/{arm}/{block}") as guard:
        guard()
        manifest = check(MODELS / arm, arm)
        start = monotonic()
        assistant = open_assistant(
            argparse.Namespace(checkpoint=MODELS / arm, tool_format="qwen")
        )
        mx.eval(assistant._tool_model.parameters())
        load_seconds = monotonic() - start
        for case in cohort["cases"]:
            if case["block"] != block:
                continue
            path = LOCAL / "traces" / arm / (case["id"] + ".json.gz")
            if path.exists():
                old = unpack(path)
                if old["cohort"] != digest(LOCAL / "cohort.json") or old[
                    "model"
                ] != fingerprint(manifest["files"]):
                    raise ValueError("Source or model differs in existing local trace")
                continue
            rows = []
            with ExitStack() as stack:
                tools = providers(case["kind"], LOCAL, stack)
                chat = assistant.conversation(tools)
                for i, q in enumerate(case["prompts"]):
                    before = list(chat.messages)
                    opening = fingerprint(
                        prompt_ids(
                            assistant.tokenizer,
                            [*before, {"role": "user", "content": q}],
                            tools,
                        )
                    )
                    if i == 0 and opening != case["opening"]:
                        raise ValueError("Mismatched initial prompt")
                    stamps = []
                    tick = monotonic()

                    def progress(pos):
                        guard()
                        stamps.append([pos, monotonic() - tick])

                    mx.reset_peak_memory()
                    trace = chat.respond(
                        q,
                        max_tokens=512,
                        max_turns=3,
                        progress=progress,
                        reuse_prefix=False,
                        prefill_cache_state=True,
                        record_kv_history=True,
                    )
                    elapsed = monotonic() - tick
                    assert fingerprint(trace["turns"][0]["prompt_ids"]) == opening
                    rows.append(
                        {
                            "trace": trace,
                            "opening": opening,
                            "before": before,
                            "after": list(chat.messages),
                            "seconds": elapsed,
                            "first_visible": first_visible(
                                trace["turns"], stamps, assistant.tokenizer
                            ),
                            "timing": stamps,
                            "peak": mx.get_peak_memory(),
                        }
                    )
            packed(
                path,
                {
                    "arm": arm,
                    "cohort": digest(LOCAL / "cohort.json"),
                    "model": fingerprint(manifest["files"]),
                    "pid": os.getpid(),
                    "load_seconds": load_seconds,
                    "rows": rows,
                },
            )
        guard(force=True)


def panel():
    prepare()
    for block, arms in enumerate((("base", "loop"), ("loop", "base"))):
        for arm in arms:
            dispatch("local_workflows", "collect", "--arm", arm, "--block", block)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "collect", "panel"))
    parser.add_argument("--arm", choices=("loop", "base"))
    parser.add_argument("--block", type=int, choices=(0, 1))
    a = parser.parse_args()
    if a.action == "prepare":
        prepare()
    elif a.action == "collect":
        collect(a.arm, a.block)
    else:
        panel()
