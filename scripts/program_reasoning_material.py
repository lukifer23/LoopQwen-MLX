"""Construct verified calculator programs from human solutions and reserve unseen problems."""

import ast
import gzip
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path
from urllib.request import urlopen

from loopqwen.calculator_tool import calculate, execute, tool
from loopqwen.evaluation.answers import numeric
from loopqwen.evaluation.program_questions import INSTRUCTION
from loopqwen.io import digest, read, save
from loopqwen.tool_session import compact, initial_messages

PRIOR = "f68fe2c"
ROOT = Path("data/research_external")
GSM_REV = "3101c7d5072418e28b9008a6636bde82a006892c"
REG = Path("research/data")
ASDIV_URL = "https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/883f90a9a65bf00304ba8f37423910fe743abc47/dataset/ASDiv.xml"
ASDIV_SHA = "ef8904068482919ac48c8eeaaf6df344b8a308ba66d048c2d4d87eab82dc4929"


def norm(s):
    return re.findall(r"[\w]+", s.casefold())


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def former(rows, prior=PRIOR):
    keys = {tuple(norm(q)[:8]) for q in rows}
    blocked = set()

    def visit(obj):
        if isinstance(obj, str):
            words = norm(obj)
            for i in range(len(words) - 7):
                k = tuple(words[i : i + 8])
                if k in keys:
                    blocked.add(k)
        elif isinstance(obj, dict):
            for v in obj.values():
                visit(v)
        elif isinstance(obj, list) and obj and not isinstance(obj[0], (int, float)):
            for v in obj:
                visit(v)

    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", prior, "research/data"], text=True
    ).splitlines()
    for f in names:
        if f.endswith(".json"):
            visit(json.loads(subprocess.check_output(["git", "show", prior + ":" + f])))
    old = set().union(
        *json.loads(
            gzip.decompress(
                Path("research/registry/historical-text-ids.json.gz").read_bytes()
            )
        ).values()
    )
    return {
        q
        for q in rows
        if tuple(norm(q)[:8]) in blocked
        or any(
            sha("text:" + " ".join((q + s).split())) in old
            for s in (
                "",
                " Answer:",
                " Show your steps. End with Final answer: <number>. Solution:",
            )
        )
    }


def program(solution):
    raw, gold = solution.rsplit("####", 1)
    annotated = re.findall(r"<<([^<>=]+)=([^<>=]+)>>", raw)
    if not 1 <= len(annotated) <= 6:
        return None
    steps, known = [], {}
    for text, v in annotated:
        text = text.strip().replace(",", "")
        try:
            result = Fraction(v.strip().replace(",", ""))
            if execute([text])[0] != result:
                return None
            # Reuse an earlier computed value only when its literal occurs as an operand.
            tree = ast.parse(text, mode="eval")

            class Refer(ast.NodeTransformer):
                def visit_Constant(self, node):
                    n = Fraction(str(node.value))
                    return ast.Name(known[n], ctx=ast.Load()) if n in known else node

            step = ast.unparse(Refer().visit(tree)).replace(" ", "")
            steps.append(step)
            if execute(steps)[-1] != result:
                return None
            known[result] = "r" + str(len(steps))
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
            return None
    return steps if str(result) == numeric(gold) else None


def source(name, url, checksum):
    path = ROOT / name
    if not path.exists():
        with urlopen(url, timeout=60) as resp:
            path.write_bytes(resp.read())
    if digest(path) != checksum:
        raise ValueError("Source digest mismatch: " + name)
    return path


def main():
    raw = read("research/experiments/gsm-equations-01.json")["source"]
    sv = read("research/experiments/checked-rationale-01.json")["sources"]["svamp"]
    url = f"https://raw.githubusercontent.com/openai/grade-school-math/{GSM_REV}/grade_school_math/data/"
    tr = source("gsm-train.jsonl", url + "train.jsonl", raw["sha256"])
    ad = source("asdiv.xml", ASDIV_URL, ASDIV_SHA)
    sw = source(sv["file"], sv["url"], sv["sha256"])
    train = [json.loads(l) for l in tr.read_text().splitlines()]
    data = {
        "ASDiv": [
            (
                " ".join(
                    (r.findtext("Body", "") + " " + r.findtext("Question", "")).split()
                ),
                numeric(re.sub(r"\s*\([^()]*\)\s*$", "", r.findtext("Answer", ""))),
            )
            for r in ET.parse(ad).iter("Problem")
        ],
        "SVAMP": [
            (r["Body"].strip() + " " + r["Question"].strip(), numeric(str(r["Answer"])))
            for r in read(sw)
        ],
    }
    questions = {w["question"] for w in train} | {
        q for group in data.values() for q, _ in group
    }
    excluded = former(questions)
    # SVAMP is derived from ASDiv. Remove overlapping sentence openings across corpora.
    sv_openings = {tuple(norm(q)[:8]) for q, _ in data["SVAMP"]}
    data["ASDiv"] = [
        (q, g) for q, g in data["ASDiv"] if tuple(norm(q)[:8]) not in sv_openings
    ]
    train_questions = {r["question"] for r in train}
    enrolled = set()
    items = []
    for src, group in data.items():
        count = 0
        for q, gold in sorted(group, key=lambda t: sha("program-eval-7921:" + t[0])):
            opening = tuple(norm(q)[:8])
            if (
                q in excluded
                or gold is None
                or q in train_questions
                or opening in enrolled
            ):
                continue
            enrolled.add(opening)
            items.append(
                {
                    "id": sha(q),
                    "question": q,
                    "source": src,
                    "gold": gold,
                    "user_request": q + INSTRUCTION,
                }
            )
            excluded.add(q)
            count += 1
            if count == 24:
                break
        assert count == 24
    items.sort(key=lambda r: sha("order:" + r["id"]))
    examples = []
    for w in sorted(train, key=lambda w: sha("program-train-2158:" + w["question"])):
        q = w["question"]
        if q in excluded:
            continue
        steps = program(w["answer"])
        if not steps:
            continue
        excluded.add(q)
        call = compact({"name": "calculate", "arguments": {"steps": steps}})
        obs = compact(calculate(steps))
        msgs = initial_messages(q + INSTRUCTION, [tool()])
        key = sha(q)
        examples.append(
            {"id": key + "-call", "messages": msgs, "visible": call, "thinking": ""}
        )
        examples.append(
            {
                "id": key + "-result",
                "messages": msgs
                + [
                    {"role": "assistant", "content": call},
                    {"role": "tool", "content": obs},
                ],
                "visible": str(execute(steps)[-1]),
                "thinking": "",
            }
        )
        if len(examples) == 1280:
            break
    assert len(examples) == 1280
    previous = read(REG / "tool-dialogue-typed-training.json")["examples"]
    # Preserve general argument/observation supervision as a rehearsal corpus.
    examples += sorted(previous, key=lambda w: sha("program-rehearsal:" + w["id"]))[
        :512
    ]
    sources = {
        "gsm_train": digest(tr),
        "asdiv": digest(ad),
        "svamp": digest(sw),
        "exclusions_git": PRIOR,
    }
    save(
        REG / "tool-program-reasoning-training.json",
        {"sources": sources, "examples": examples},
    )
    save(REG / "tool-program-reasoning-01.json", {"sources": sources, "rows": items})
    print(len(examples), "supervised turns;", len(items), "fresh reasoning questions")


if __name__ == "__main__":
    main()
