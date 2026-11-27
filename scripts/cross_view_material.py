"""Freeze SQuAD teaching views and independent Quoref questions with real CSV callbacks."""

import csv
import io
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from loopqwen import qwen_tools
from loopqwen.csv_tools import tools
from loopqwen.evaluation.extractive import normalized
from loopqwen.export_runtime import load_export_tokenizer
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.tool_session import compact

NAME = "cross-view-reading-01"
SPEC = Path("research/experiments") / (NAME + ".json")


def select(articles, n, seed, words, exclude):
    def key(x):
        return fingerprint([seed, x])

    chosen = []
    for art in sorted(articles, key=lambda a: key(a["title"])):
        title = art["title"].replace("_", " ")
        if normalized(title) in exclude:
            continue
        for para in sorted(art["paragraphs"], key=lambda p: key(p["context"])):
            passage = para["context"]
            if not 40 <= len(passage.split()) <= words:
                continue
            questions = []
            for q in sorted(para["qas"], key=lambda q: key(q["id"])):
                answers = {a["text"] for a in q["answers"]}
                if len(answers) != 1:
                    continue
                answer = answers.pop()
                if (
                    not normalized(answer)
                    or not 1 <= len(answer.split()) <= 12
                    or any(
                        passage[a["answer_start"] : a["answer_start"] + len(answer)]
                        != answer
                        for a in q["answers"]
                    )
                ):
                    continue
                gs = normalized(answer)
                if any(
                    re.search(
                        r"(?<!\w)" + re.escape(gs) + r"(?!\w)", normalized(c["answer"])
                    )
                    or re.search(
                        r"(?<!\w)" + re.escape(normalized(c["answer"])) + r"(?!\w)", gs
                    )
                    for c in questions
                ):
                    continue
                questions.append(
                    {
                        "id": q["id"],
                        "question": q["question"],
                        "answer": answer,
                        "offset": q["answers"][0]["answer_start"],
                    }
                )
                if len(questions) == 2:
                    break
            if len(questions) == 2:
                chosen.append(
                    {
                        "page": title,
                        "url": art.get("url"),
                        "passage": passage,
                        "questions": questions,
                    }
                )
                exclude.add(normalized(title))
                break
        if len(chosen) == n:
            return chosen
    raise ValueError("Insufficient independent eligible publisher articles")


def views(group, neighbors, index, directory):
    filename, keycol, textcol = (
        ("notes.csv", "title", "passage"),
        ("sources.csv", "entry", "text"),
    )[index % 2]
    target = Path(directory) / filename
    with target.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([keycol, textcol])
        writer.writerows(
            (a["page"], a["passage"])
            for a in sorted([group, *neighbors], key=lambda x: fingerprint(x["page"]))
        )
    doc = target.read_bytes().decode()
    provider = tools([target], keyed=True)
    defs = qwen_tools.declarations(provider)
    key = group["page"]
    arguments = {
        "document": filename,
        "key_column": keycol,
        "key": key,
        "value_column": textcol,
    }
    call = {"name": "csv_lookup", "arguments": arguments}
    obs = provider[0].callback(**arguments)
    assert obs == {"value": group["passage"]}
    for q in group["questions"]:
        instruction = "Give the short answer to the question."
        inline = [
            {
                "role": "user",
                "content": group["passage"]
                + "\n\n"
                + q["question"]
                + "\n"
                + instruction,
            }
        ]
        opening = [
            {
                "role": "user",
                "content": f'Use the attached file to read the entry for "{key}".\n'
                + q["question"]
                + "\n"
                + instruction,
            }
        ]
        action = qwen_tools.assistant_message([call])
        file = [*opening, action, {"role": "tool", "content": compact(obs)}]
        yield (
            q,
            {
                "inline": inline,
                "file": file,
                "opening": opening,
                "action": action,
                "functions": defs,
                "document": filename,
                "csv": doc,
                "call": call,
            },
        )


def build():
    s = read(SPEC)
    raw = Path("data/research_external")
    source = raw / "quoref/train-dev.zip"
    squad = raw / "squad-train-v1.1.json"
    assert (
        digest(source) == s["source_hashes"]["quoref"]
        and digest(squad) == s["source_hashes"]["squad"]
    )
    with ZipFile(source) as z:
        # Only the publisher development file enters this experiment.
        dev = json.loads(z.read("quoref-train-dev-v0.1/quoref-dev-v0.1.json"))["data"]
    used = set()
    held = select(dev, s["test_articles"], s["sample_seed"], 500, used)
    train = select(
        read(squad)["data"], s["train_articles"], s["sample_seed"], 300, used
    )

    def texts(articles):
        return {
            normalized(t)
            for a in articles
            for t in [a["passage"], *(q["question"] for q in a["questions"])]
        }

    assert not texts(train) & texts(held)
    release = Path(s["checkpoint"])
    tok = load_export_tokenizer(
        release,
        eos_token_ids=[read(release / "config.json")["text_config"]["eos_token_id"]],
    )
    examples, rows, documents = [], [], {}
    with TemporaryDirectory() as tmp:
        for teaching, articles in ((True, train), (False, held)):
            for i, g in enumerate(articles):
                neigh = [articles[(i + n) % len(articles)] for n in (1, 2)]
                for q, v in views(g, neigh, i, tmp):
                    if teaching:
                        target = {"role": "assistant", "content": q["answer"]}
                        for kind, decl, history, reply in (
                            ("inline", [], v["inline"], target),
                            ("file", v["functions"], v["file"], target),
                            ("opening", v["functions"], v["opening"], v["action"]),
                        ):
                            examples.append(
                                {
                                    "id": q["id"] + "-" + kind,
                                    "source_question": q["id"],
                                    "source_article": g["page"],
                                    "kind": kind,
                                    "tools": decl,
                                    "messages": history,
                                    "target": reply,
                                }
                            )
                    else:
                        for family in ("inline", "file"):
                            is_file = family == "file"
                            body = (v["opening"] if is_file else v["inline"])[0][
                                "content"
                            ]
                            folder = fingerprint(g["page"])[:12]
                            if is_file:
                                documents.setdefault(
                                    folder + "/" + v["document"], v["csv"]
                                )
                            rows.append(
                                {
                                    "id": q["id"] + "-" + family,
                                    "question_id": q["id"],
                                    "family": family,
                                    "cluster": g["page"],
                                    "questions": [body],
                                    "gold": q["answer"],
                                    "other": next(
                                        x["answer"] for x in g["questions"] if x != q
                                    ),
                                    "source": g["passage"],
                                    "document": folder + "/" + v["document"]
                                    if is_file
                                    else None,
                                    "calls": [v["call"]] if is_file else [],
                                    "observation": {"value": g["passage"]}
                                    if is_file
                                    else None,
                                    "initial_ids": tok.encode(
                                        qwen_tools.render(
                                            tok,
                                            [{"role": "user", "content": body}],
                                            v["functions"] if is_file else [],
                                        ),
                                        add_special_tokens=False,
                                    ),
                                }
                            )
    rehearsal = read("research/data/role-binding-training.json")["examples"]
    reuse = [r for r in rehearsal if r["id"].startswith("wiki-train")][: s["rehearsal"]]
    assert len(reuse) == s["rehearsal"]
    examples.extend({**r, "kind": "rehearsal"} for r in reuse)
    past = read("research/data/role-binding-panel.json")
    seen = set()
    for row in sorted(
        (w for w in past["rows"] if w["family"] == "table"),
        key=lambda w: fingerprint([s["sample_seed"], w["id"]]),
    ):
        if row["cluster"] in seen:
            continue
        seen.add(row["cluster"])
        documents[row["document"]] = past["documents"][row["document"]]
        rows.append(
            {
                **row,
                "family": "retention",
                "calls": [{"name": "csv_lookup", "arguments": row["arguments"]}],
                "observation": {"value": row["gold"]},
                "other": None,
            }
        )
        if len(seen) == s["retain_tables"]:
            break
    assert len(seen) == s["retain_tables"]
    rows.sort(key=lambda w: fingerprint([s["sample_seed"], w["id"]]))
    for k, payload in (
        ("training", {"articles": train, "examples": examples}),
        ("material", {"articles": held, "rows": rows, "documents": documents}),
    ):
        path = Path(s[k])
        if path.exists():
            assert read(path) == payload
        else:
            save(path, payload)
    print(len(train), len(held), len(examples), len(rows))


if __name__ == "__main__":
    build()
