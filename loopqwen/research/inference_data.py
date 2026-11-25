"""Deterministic counterfactual inference triplets from the publisher training and dev splits."""

import csv
import itertools
import json
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from mlx_lm.utils import load_tokenizer

from ..identity import fingerprint
from ..io import digest
from ..model_loader import local_snapshot
from . import core, projected_distillation

LABELS = {"entailment": "yes", "contradiction": "no", "neutral": "maybe"}
SOURCE = Path("data/research_external")


@contextmanager
def entries(s, corpus, train):
    name = (
        ("esnli_train_1.csv" if train else "esnli_dev.csv")
        if corpus == "esnli"
        else "multinli_1.0.zip"
    )
    spec = s["files"][name]
    path = SOURCE / name
    if not path.is_file():
        from shutil import copyfileobj

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".download")
        with urlopen(spec["url"], timeout=90) as resp, tmp.open("wb") as stream:
            copyfileobj(resp, stream)
        if digest(tmp) != spec["sha256"]:
            raise ValueError("Publisher download differs")
        tmp.replace(path)
    if digest(path) != spec["sha256"]:
        raise ValueError("Publisher file differs")
    if corpus == "esnli":
        with path.open(newline="") as f:
            yield (
                (
                    r["Sentence1"],
                    r["Sentence2"],
                    r["gold_label"],
                    r["Explanation_1"].strip(),
                    r["pairID"],
                    "caption",
                )
                for r in csv.DictReader(f)
            )
    else:
        split = "train" if train else "dev_mismatched"
        with (
            ZipFile(path) as z,
            z.open("multinli_1.0/multinli_1.0_" + split + ".jsonl") as f,
        ):
            yield (
                (
                    r["sentence1"],
                    r["sentence2"],
                    r["gold_label"],
                    "",
                    r["pairID"],
                    r["genre"],
                )
                for r in map(json.loads, f)
            )


def prompt(tokenizer, a, h, template):
    if template == 0:
        user = f"Premise: {a}\nClaim: {h}\nDoes the premise imply the claim? Answer yes if it follows, no if it contradicts the premise, or maybe if the premise leaves it open."
    elif template == 1:
        user = f"Read this statement: {a}\nSomeone concludes: {h}\nIs that conclusion guaranteed (yes), ruled out (no), or uncertain (maybe)?"
    elif template == 2:
        user = f"Assume the following is true. {a}\nCan we conclude that {h}\nUse yes for entailed, no for contradicted and maybe for neither."
    else:
        user = f"Passage: {a}\nProposed inference: {h}\nDecide whether the inference must be true, must be false, or could go either way. Give your final judgment as yes, no, or maybe, respectively."
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def groups(s, corpus, train, used, count):
    table = {}
    with entries(s, corpus, train) as source:
        for a, h, label, reason, uid, genre in itertools.islice(
            source, s["train_scan"] if train else None
        ):
            if (
                label not in LABELS
                or not a
                or not h
                or (train and corpus == "esnli" and not reason)
            ):
                continue
            a, h = " ".join(a.split()), " ".join(h.split())
            if len(a) + len(h) > s["character_limit"]:
                continue
            key = a.casefold()
            if key in used:
                continue
            row = {
                "premise": a,
                "hypothesis": h,
                "label": LABELS[label],
                "reason": reason,
                "source": corpus,
                "source_id": uid,
                "genre": genre,
                "id": fingerprint([a, h]),
            }
            table.setdefault(key, {}).setdefault(label, row)
    options = [k for k, v in table.items() if len(v) == 3]
    options.sort(key=lambda k: fingerprint([s["seed"], corpus, train, k]))
    if len(options) < count:
        raise ValueError(
            f"Insufficient {corpus} distinct complete triplets ({len(options)})"
        )
    selected = options[:count]
    used.update(selected)
    return [
        sorted(table[k].values(), key=lambda r: fingerprint([s["seed"], r["id"]]))
        for k in selected
    ]


def material(s, excluded=()):
    tok = load_tokenizer(local_snapshot())
    used = {r["premise"].casefold() for r in excluded}
    sources = ("esnli", "multinli")
    evaluation = {c: groups(s, c, False, used, s["per_source_groups"]) for c in sources}
    panel = []
    for i in range(s["per_source_groups"]):
        for c in sources:
            for r in evaluation[c][i]:
                row = {k: v for k, v in r.items() if k != "reason"}
                txt = prompt(tok, r["premise"], r["hypothesis"], s["eval_template"])
                panel.append(
                    dict(**row, prompt_ids=tok.encode(txt, add_special_tokens=False))
                )
    return panel


def training(s, panel):
    tok = load_tokenizer(local_snapshot())
    used = {r["premise"].casefold() for r in panel}
    groups_by = {
        c: groups(s, c, True, used, s["updates"]) for c in ("esnli", "multinli")
    }
    anchors = projected_distillation.anchors(tok, s)
    docs = []
    keys = []
    for i, anchor in enumerate(anchors):
        for c in groups_by:
            for r in groups_by[c][i]:
                txt = prompt(tok, r["premise"], r["hypothesis"], i % 3)
                label = r["label"]
                reasoning = ("My judgment is " + label + ". " + r["reason"]).strip()
                doc = core.doc(
                    tok, txt, reasoning + "\n</think>\n\n" + label.capitalize() + "."
                )
                if len(doc["ids"]) - 1 > s["cap_positions"]:
                    raise ValueError(
                        "Registered training cap too small for selected example"
                    )
                docs.append(doc)
                keys.append(r["id"])
        docs.append({"prefix": anchor["prefix"], "ids": anchor["ids"]})
    return docs, {
        "ordered_questions": keys,
        "documents": fingerprint(docs),
        "anchor_sha256": fingerprint(anchors),
        "response_tokens": sum(len(d["ids"]) - d["prefix"] for d in docs),
        "premises": fingerprint(
            sorted(used - {r["premise"].casefold() for r in panel})
        ),
    }
