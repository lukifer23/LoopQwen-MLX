"""Fit a small input-only policy for existing, independently qualified recurrent decoders."""

import csv
import itertools
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn

from ..context_policy import features
from ..evaluation.language7 import prompt_ids
from ..identity import fingerprint
from ..io import digest, read, save
from . import inference_data, memory_data

LABELS = ["ordinary", "arithmetic", "register", "entailment"]


def chat(t, text):
    return t.encode(
        t.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        ),
        add_special_tokens=False,
    )


def inputs(t, s):
    count = s["route_examples_per_class"]
    files = s["routing_sources"]
    for name, expected in files.items():
        if digest(Path(name)) != expected:
            raise ValueError("Router training source differs")
    gsm = Path("data/research_external/gsm-train.jsonl")
    allowed = set(
        read("research/data/all-layer-reasoning-training.json")["ordered_questions"]
    )
    questions = sorted(
        (
            q
            for line in gsm.read_bytes().splitlines()
            if fingerprint(q := " ".join(json.loads(line)["question"].split()))
            in allowed
        ),
        key=lambda q: fingerprint([s["seed"], q]),
    )[:count]
    with Path("data/research_external/esnli_train_1.csv").open(newline="") as f:
        pairs = [
            (r["Sentence1"], r["Sentence2"])
            for r in itertools.islice(csv.DictReader(f), count)
        ]
    articles = read("data/research_external/squad-train-v1.1.json")["data"]
    passages = [
        (p["context"], p["qas"][0]["question"])
        for a in articles
        for p in a["paragraphs"]
        if p["qas"]
    ]
    passages.sort(key=lambda v: fingerprint([s["seed"], v]))
    examples = []
    for j, (q, (premise, claim)) in enumerate(zip(questions, pairs, strict=True)):
        passage, question = passages[j]
        text = (
            passage + "\n\n" + question
            if j % 3 == 0
            else "Summarize this passage in your own words.\n" + passage
            if j % 3 == 1
            else question
        )
        ordinary = chat(t, text)
        math = (
            prompt_ids(t, {"question": q}, {"demos": 0}) if j % 2 == 0 else chat(t, q)
        )
        register = memory_data.make_item(
            t,
            s["seed"] + j,
            768 + 384 * (j % 4),
            0.1 + 0.2 * (j % 4),
            ("table", "statement", "note")[j % 3],
        )["prompt_ids"]
        nli = t.encode(
            inference_data.prompt(t, premise, claim, j % 3), add_special_tokens=False
        )
        examples.extend([ordinary, math, register, nli])
    if len(examples) != 4 * count:
        raise ValueError("Incomplete route training cohort")
    return examples


def qualified_expert(name, info):
    """Check both predeclared outcomes and their separately sealed content reviews."""
    for stage in (info["qualification"], info["confirmation"]):
        archive = Path(stage["directory"])
        annotation = Path(stage["audit"])
        if (
            digest(archive / "checksums.json") != stage["inventory"]
            or not read(archive / "decision.json")["passed"]
            or digest(annotation) != stage["audit_sha256"]
        ):
            raise ValueError("Missing measured specialist prerequisite")
        if name == "arithmetic":
            from scripts.research_answer_audit import score

            audited = [score(archive.name, read(annotation))]
        elif name == "entailment":
            from scripts.research_relation_audit import audit

            audited = audit(archive.name, read(annotation))["scores"].values()
        else:
            raise ValueError("Unknown expert")
        if any(r["wins"] <= r["losses"] for r in audited):
            raise ValueError("The independent content review did not pass")
    primary = read(Path(info["qualification"]["directory"]) / "decision.json")
    confirmation = read(Path(info["confirmation"]["directory"]) / "decision.json")
    if (
        confirmation["qualification"] != info["qualification"]["inventory"]
        or primary["weights"] != info["weights"]
        or confirmation["weights"] != info["weights"]
    ):
        raise ValueError("Different specialist or qualification at replication")


def package(s, model, tokenizer, examples, root, budget):
    directory = root / "model"
    parent = Path(s["reader"])
    shutil.copytree(parent, directory)
    files = [
        "adapters.safetensors",
        "attention-reader.safetensors",
        "attention-reader.json",
    ]
    for name, info in s["experts"].items():
        source = Path(info["model"])
        qualified_expert(name, info)
        if digest(source / "adapters.safetensors") != info["weights"]:
            raise ValueError("Wrong specialist checkpoint")
        names = ["adapters.safetensors"]
        if name == "arithmetic":
            names += ["deliberation.json", "exit-readout.safetensors"]
        local = directory / name
        local.mkdir()
        for f in names:
            shutil.copyfile(source / f, local / f)
            files.append(name + "/" + f)
    selector = s.get("selector_source")
    if selector:
        arc = Path(selector["directory"])
        source = arc / "model/selector.safetensors"
        report = read(arc / "router-training.json")
        if (
            digest(arc / "checksums.json") != selector["inventory"]
            or digest(source) != selector["sha256"]
            or not read(arc / "decision.json")["passed"]
            or report["documents"] != fingerprint(examples)
            or report["n"] != len(examples)
        ):
            raise ValueError("Qualified input selector or training cohort changed")
        shutil.copyfile(source, directory / "selector.safetensors")
        save(root / "router-training.json", dict(report, source=selector))
    else:
        fit_selector(s, model, examples, directory, root, budget)
    files.append("selector.safetensors")
    save(
        directory / "context-policy.json",
        {
            "schema": 2,
            "labels": LABELS,
            "confidence": s["confidence"],
            "experts": list(s["experts"]),
            "files": {f: digest(directory / f) for f in files},
        },
    )
    return digest(directory / "context-policy.json")


def fit_selector(s, model, examples, directory, root, budget):
    classifier = nn.Linear(2 * model.args.hidden_size, len(LABELS))
    optimizer = optim.Adam(learning_rate=s["route_lr"])
    embed = model.text.model.embed_tokens
    vectors = []
    for row in examples:
        budget.check()
        v = features(embed, row)
        mx.eval(v)
        vectors.append(v)
    x = mx.stack(vectors)
    y = mx.tile(mx.arange(len(LABELS)), s["route_examples_per_class"])
    vg = nn.value_and_grad(
        classifier, lambda m: nn.losses.cross_entropy(m(x), y).mean()
    )
    history = []
    for _ in range(s["route_updates"]):
        budget.check()
        value, grad = vg(classifier)
        optimizer.update(classifier, grad)
        mx.eval(value, classifier.parameters(), optimizer.state)
        history.append(value.item())
    classifier.save_weights(str(directory / "selector.safetensors"))
    save(
        root / "router-training.json",
        {
            "documents": fingerprint(examples),
            "n": len(examples),
            "history": history,
            "training_correct": (classifier(x).argmax(axis=-1) == y).sum().item(),
        },
    )
