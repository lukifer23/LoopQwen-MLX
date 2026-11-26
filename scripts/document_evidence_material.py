"""Enroll real searches, returned passages and short answers for the recurrent document reader."""

import hashlib
import random
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen.document_tools import documents
from loopqwen.io import read, save
from loopqwen.tool_session import compact, initial_messages
from scripts.document_search_material import (
    DATA,
    SOURCES,
    article,
    build,
    normalized,
    source,
)

ROOT = Path("research/data")
REVISION = "5bd60c2"
GUIDES = (
    (
        "PHASE3_DATA",
        "How many procedural instances are in the V2 manifest?",
        "24,000",
        "24,000 procedural instances",
    ),
    ("PHASE3_PLAN", "Which training seed did Phase 3 register?", "73", "Use seed 73"),
    (
        "PHASE4_ARCHITECTURE",
        "How many parameters does block LoRA add?",
        "1,336,320",
        "Block LoRA adds 1,336,320 parameters",
    ),
    (
        "PHASE4_DATA",
        "How many independent passages are in the language-retention sample?",
        "120",
        "120 independent passages",
    ),
    (
        "PHASE5_PLAN",
        "How many updates initially alternate between depths one and two?",
        "48",
        "first 48 updates alternate",
    ),
    (
        "PHASE5_RESULTS",
        "What was the three-hop score after shuffling the displayed edges?",
        "36/64",
        "correctness to 36/64",
    ),
    (
        "PHASE6_RESULTS",
        "What was the minimum number correct required by the registered canonical floor?",
        "365",
        "floor required 365 correct",
    ),
    (
        "PHASE7_PLAN",
        "How many native readouts did the published R480 reproduce?",
        "48",
        "all 48 native readouts",
    ),
)


def guides():
    docs, rows = [], []
    for name, question, gold, quotation in GUIDES:
        key = "docs/" + name + ".md"
        src = subprocess.check_output(["git", "show", REVISION + ":" + key])
        text = src.decode()
        assert text.count(quotation) == 1 and quotation.count(gold) == 1
        pos = text.index(quotation) + quotation.index(gold)
        filename = Path(key).name
        docs.append(
            dict(
                source="Project guides",
                key=key,
                revision=REVISION,
                title=name,
                document=filename,
                sha256=hashlib.sha256(src).hexdigest(),
                characters=len(text),
            )
        )
        rows.append(
            dict(
                source="Project guides",
                id=name,
                document=filename,
                question=question,
                request=f"Check {filename}. {question} Give just the answer.",
                refs=[gold],
                answer_ranges=[[pos, pos + len(gold)]],
            )
        )
    return docs, rows


def training_documents(blocked, rng):
    bank = read(source(SOURCES["SQuAD", "train"]))["data"]
    rng.shuffle(bank)
    docs = []
    for a in bank:
        if normalized(a["title"].replace("_", " ")) in blocked:
            continue
        ps = list(a["paragraphs"])
        rng.shuffle(ps)
        selected = []
        for p in ps:
            qs = list(p["qas"])
            rng.shuffle(qs)
            for q in qs:
                gold = q["answers"][0]["text"].strip()
                if 1 <= len(gold.split()) <= 10 and gold in p["context"]:
                    selected.append((q["id"], q["question"], gold, p["context"]))
                    break
            if len(selected) == 8:
                break
        if len(selected) == 8:
            docs.append(("SQuAD train", a["title"], article("SQuAD", a), selected))
        if len(docs) == 48:
            break
    assert len(docs) == 48
    bank = read(source(SOURCES["QuAC", "train"]))["data"]
    rng.shuffle(bank)
    titles = set(blocked)
    for a in bank:
        name = normalized(a["title"])
        if name in titles:
            continue
        for p in a["paragraphs"]:
            q = p["qas"][0]
            answer = q["orig_answer"]["text"].strip()
            body = p["context"].removesuffix(" CANNOTANSWER").strip()
            if (
                answer == "CANNOTANSWER"
                or not 1 <= len(answer.split()) <= 12
                or answer not in body
            ):
                continue
            docs.append(
                (
                    "QuAC train",
                    a["title"],
                    body,
                    [(q["id"], q["question"].strip(), answer, body)],
                )
            )
            titles.add(name)
            break
        if len(docs) == 304:
            break
    assert len(docs) == 304
    rng.shuffle(docs)
    return docs


def training(held, old):
    rng = random.Random(98561)
    blocked = {
        normalized(d["title"].replace("_", " "))
        for m in [held, old]
        for d in m["documents"]
    }
    bank = training_documents(blocked, rng)
    held_titles = {
        (family, title)
        for family, count in (("SQuAD train", 4), ("QuAC train", 16))
        for title in [d[1] for d in bank if d[0] == family][:count]
    }
    examples, episodes, validation = [], [], []
    for start in range(0, len(bank), 16):
        group = bank[start : start + 16]
        with TemporaryDirectory() as tmp:
            files = []
            for i, (_, title, body, _) in enumerate(group, start):
                slug = re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-")
                # Filenames are independent of the question and the gold span.
                name = (
                    f"{slug}-{i}.txt"
                    if i % 3 == 0
                    else (
                        f"notes-{i}.md"
                        if i % 3 == 1
                        else f"{rng.randrange(1000, 10000)}-{i:05d}.txt"
                    )
                )
                path = Path(tmp) / name
                path.write_bytes(body.encode())
                files.append(path)
            with documents(files) as tools:
                for (family, title, body, questions), path in zip(
                    group, files, strict=True
                ):
                    for ident, question, gold, evidence in questions:
                        wording = rng.choice(
                            [
                                f"Look in {path.name}. {question} Answer briefly.",
                                f"Use {path.name} to answer this question: {question}",
                                f"{question} Please check {path.name} and give the short answer.",
                            ]
                        )
                        opening = initial_messages(wording, tools)

                        def event(query):
                            call = compact(
                                {
                                    "name": "search_documents",
                                    "arguments": {
                                        "query": query,
                                        "document": path.name,
                                    },
                                }
                            )
                            obs = tools[0].callback(query=query, document=path.name)
                            return call, obs

                        call, obs = event(question)
                        ranges = [
                            (o.start() + m.start(), o.start() + m.end())
                            for o in re.finditer(re.escape(evidence), body)
                            for m in re.finditer(re.escape(gold), evidence)
                        ]
                        labels = {
                            str(i + 1): any(
                                h["start"] <= a < b <= h["stop"] for a, b in ranges
                            )
                            for i, h in enumerate(obs["passages"])
                        }
                        located = any(labels.values())
                        reserved = (family, title) in held_titles

                        def add(suffix, messages, target):
                            row = dict(
                                id=ident + suffix,
                                source=family,
                                kind="read" if suffix == "-read" else "search",
                                messages=list(messages),
                                thinking="",
                                visible=target,
                            )
                            if suffix == "-read":
                                row["known"] = labels
                            if reserved:
                                if suffix == "-read" and labels:
                                    validation.append(row)
                            else:
                                examples.append(row)

                        add("-search", opening, call)
                        messages = [
                            *opening,
                            dict(role="assistant", content=call),
                            dict(role="tool", content=compact(obs)),
                        ]
                        add(
                            "-read",
                            messages,
                            gold
                            if located
                            else "I couldn't verify the answer in the returned passages.",
                        )
                        if len(episodes) % 5 == 0:
                            wrong, empty = event(path.name)
                            if not empty["passages"]:
                                add(
                                    "-retry",
                                    [
                                        *opening,
                                        dict(role="assistant", content=wrong),
                                        dict(role="tool", content=compact(empty)),
                                    ],
                                    call,
                                )
                        episodes.append(
                            dict(
                                id=ident,
                                source=family,
                                title=title,
                                question=question,
                                document_sha256=hashlib.sha256(
                                    body.encode()
                                ).hexdigest(),
                                question_query_found=located,
                                split="validation" if reserved else "train",
                            )
                        )
    assert len(episodes) == len({e["id"] for e in episodes}) == 640
    typed = read(ROOT / "tool-dialogue-typed-training.json")["examples"]
    examples.extend(rng.sample(typed, 192))
    rng.shuffle(examples)
    return dict(
        episodes=episodes,
        examples=examples,
        validation=validation,
        sources={name: list(SOURCES[name, "train"]) for name in ("SQuAD", "QuAC")},
    )


def main():
    old = read(DATA)
    docs, rows = guides()
    out = ROOT / "document-evidence-panel.json"
    build(out, 985623, {(d["source"], d["key"]) for d in old["documents"]}, docs, rows)
    held = read(out)
    assert len({r["document"] for r in held["rows"]}) == 40
    data = training(held, old)
    save(ROOT / "document-evidence-training.json", data)
    print(
        len(data["examples"]),
        len(data["validation"]),
        "reserved observations",
        flush=True,
    )


if __name__ == "__main__":
    main()
