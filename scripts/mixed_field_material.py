"""Freeze one mixed ordinary-use stream of unseen questions for a single offline release."""

import hashlib
import random
import re
import subprocess
import xml.etree.ElementTree as ET
from importlib.metadata import version
from pathlib import Path

from bridge_reading_material import canonical, enroll
from mlx_lm.utils import load_tokenizer
from program_reasoning_material import ASDIV_SHA, ASDIV_URL, former, norm, source
from table_question_panel import select

from loopqwen.bridge_reader import messages
from loopqwen.evaluation import (
    program_questions as arithmetic,
)
from loopqwen.evaluation import (
    table_questions as table,
)
from loopqwen.evaluation.answers import numeric
from loopqwen.evaluation.mixed_field import catalog
from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.package_tools import snapshot
from loopqwen.tool_session import initial_messages, prompt

DATA = Path("research/data")
REV = "79d9293"


def order(key):
    return hashlib.sha256(("field-mix-493781:" + key).encode()).digest()


def main():
    tok = load_tokenizer(local_snapshot())
    old = []
    for split, field in [("train", "examples"), ("validation", "rows")]:
        old.extend(read(DATA / f"bridge-reading-{split}.json")[field])
    excluded = {
        txt
        for r in old
        for txt in [
            canonical(r["question"]),
            *[canonical(r["titles"][i]) for i in r["support"]],
        ]
    }
    reading = enroll("validation", 10, excluded, tok)
    rows = []
    for r in reading["rows"]:
        content = messages(r["question"], r["titles"], r["articles"])[1]["content"]
        rows.append(
            dict(
                id=r["id"],
                family="reading",
                question="Give a short answer using the documents.\n" + content,
                gold=r["gold"],
            )
        )
    path = source("asdiv.xml", ASDIV_URL, ASDIV_SHA)
    questions = [
        dict(
            question=" ".join(
                (r.findtext("Body", "") + " " + r.findtext("Question", "")).split()
            ),
            answer=re.sub(r"\s*\([^()]*\)\s*$", "", r.findtext("Answer", "")),
        )
        for r in ET.parse(path).iter("Problem")
    ]
    blocked = former({r["question"] for r in questions})
    for name in ("tool-program-reasoning-training", "tool-program-reasoning-01"):
        d = read(DATA / (name + ".json"))
        for e in d.get("examples", d.get("rows")):
            if "question" in e:
                blocked.add(e["question"])
            elif e["messages"]:
                blocked.add(
                    next(
                        m["content"] for m in e["messages"] if m["role"] == "user"
                    ).removesuffix(arithmetic.INSTRUCTION)
                )
    openings = {tuple(norm(t)[:8]) for t in blocked}
    enrolled = 0
    for r in sorted(questions, key=lambda r: order(r["question"])):
        key = tuple(norm(r["question"])[:8])
        value = numeric(r["answer"])
        if value is not None and key not in openings:
            rows.append(
                dict(
                    id="asdiv-" + order(r["question"]).hex(),
                    family="arithmetic",
                    question=r["question"] + arithmetic.INSTRUCTION,
                    gold=str(value),
                )
            )
            openings.add(key)
            enrolled += 1
            if enrolled == 10:
                break
    db = []
    for name in ("tool-table-queries-01", "tool-sql-learning-panel"):
        db.extend(r for r in read(DATA / (name + ".json"))["rows"] if "table_id" in r)
    seen = set(read(DATA / "tool-sql-learning-training.json")["training_tables"]) | {
        r["table_id"] for r in db
    }
    for r in select(
        "test",
        2,
        "field-mix-sql-76297",
        seen,
        {r["question"].casefold().strip() for r in db},
    ):
        rows.append(
            dict(
                r,
                id="sql-" + r["id"],
                family="database",
                user_request=r["question"] + table.INSTRUCTION,
            )
        )
    # These ten installed packages differ from the five historical version requests.
    earlier = read(
        read("research/results/tool-dialogue-package-chain-01/registration.json")[
            "material"
        ]
    )
    used = {
        n.casefold()
        for n in read("research/experiments/tool-language-release-01.json")["packages"]
    }
    used.update(w[k] for w in earlier["rows"] for k in ("root", "target"))
    packages = sorted(set(earlier["environment"]) - used - {"loopqwen"}, key=order)[:10]
    for i, name in enumerate(packages):
        question = [
            f"Which version of {name} is installed in this Python environment? Check locally.",
            f"Check the installed distribution {name} and tell me its version.",
        ][i % 2]
        rows.append(
            dict(
                id="dist-" + name,
                family="package",
                question=question,
                name=name,
                gold=version(name),
            )
        )
    docs = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", REV, "docs"], text=True
    ).splitlines()
    # Closed historical phase files remain reproducible when the live manuscript changes.
    docs = sorted(
        (p for p in docs if p.startswith("docs/PHASE") and p.endswith(".md")), key=order
    )[:8]
    file_info = {}
    for i, p in enumerate(docs):
        content = subprocess.check_output(["git", "show", REV + ":" + p])
        if content != Path(p).read_bytes():
            raise ValueError("Local field fixture changed")
        stats = dict(bytes=len(content), lines=len(content.splitlines()))
        file_info[p] = dict(stats, sha256=digest(Path(p)))
        unit = ["lines", "bytes"][i % 2]
        question = f"Check the project file {p}. How many {unit} does it contain?"
        rows.append(
            dict(
                id="file-" + unit + "-" + p,
                family="file",
                question=question,
                path=p,
                unit=unit,
                gold=str(stats[unit]),
            )
        )
    resources = dict(environment=snapshot(), files=file_info)
    for row in rows:
        fns = catalog(row, resources, True)
        q = row.get("user_request", row["question"])
        msgs = initial_messages(q, fns) if fns else [dict(role="user", content=q)]
        row["initial_ids"] = tok.encode(prompt(tok, msgs), add_special_tokens=False)
    random.Random(73091).shuffle(rows)
    from collections import Counter

    counts = Counter(r["family"] for r in rows)
    if len(rows) != 50 or len({r["id"] for r in rows}) != 50:
        raise ValueError(f"Insufficient distinct mixed field requests: {counts}")
    save(
        DATA / "release-mixed-field-01.json",
        dict(
            revision=REV,
            arithmetic_source=ASDIV_URL,
            arithmetic_sha256=digest(path),
            hotpot_sha256=reading["sha256"],
            environment=resources["environment"],
            files=file_info,
            rows=rows,
        ),
    )


if __name__ == "__main__":
    main()
