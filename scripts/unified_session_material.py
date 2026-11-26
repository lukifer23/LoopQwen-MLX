"""Build the jointly trained assistant's complete conversations and fresh field requests."""

import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download
from mlx_lm.utils import load_tokenizer
from shared_session_material import DATA, panel, sha, windows

from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.tool_session import prompt

PRIOR = "6bbe724"
DOLLY = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"


def user(row):
    return "\n".join(x["content"] for x in row["messages"] if x["role"] == "user")


def norm(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def main():
    tok = load_tokenizer(local_snapshot())
    field = read(DATA / "release-mixed-field-01.json")
    rows = panel(
        tok, field, prior=PRIOR, selection="balanced-independent", destination=None
    )
    path = Path(
        hf_hub_download(
            "databricks/databricks-dolly-15k",
            "databricks-dolly-15k.jsonl",
            repo_type="dataset",
            revision=DOLLY,
        )
    )
    dolly = [json.loads(line) for line in path.read_text().splitlines()]
    source = read(DATA / "shared-session-training.json")
    original = read(DATA / "tool-dialogue-typed-training.json")["examples"]
    held_contexts = set()
    queries = set()
    counts = Counter()
    for w in sorted(
        dolly,
        key=lambda r: sha("natural-service-field" + r["instruction"] + r["context"]),
    ):
        cat = w["category"]
        question = w["instruction"].strip()
        context = w["context"].strip()
        gold = w["response"].strip()
        if (
            cat not in ("closed_qa", "information_extraction")
            or counts[cat] >= 12
            or not context
        ):
            continue
        if not (
            40 <= len(tok.encode(context, add_special_tokens=False)) <= 1200
            and 1 <= len(gold.split()) <= 7
        ):
            continue
        if (
            any(c in gold for c in ("\n", ";", "?"))
            or " " + norm(gold) + " " not in " " + norm(context) + " "
        ):
            continue
        if (
            len(norm(gold)) < 2
            or norm(context) in held_contexts
            or norm(question) in queries
        ):
            continue
        q = question + "\n\n" + context + "\n\nReturn only the answer."
        rows.append(
            {
                "family": "user_extraction",
                "source": "Dolly " + cat,
                "id": sha(q),
                "question": q,
                "gold": gold,
                "initial_ids": tok.encode(
                    prompt(tok, [{"role": "user", "content": q}]),
                    add_special_tokens=False,
                ),
            }
        )
        held_contexts.add(norm(context))
        queries.add(norm(question))
        counts[cat] += 1
        if sum(counts.values()) == 24:
            break
    assert counts == {"closed_qa": 12, "information_extraction": 12}
    random.Random(934372).shuffle(rows)
    assert len(rows) == len({w["id"] for w in rows}) == 144
    save(
        DATA / "unified-session-panel.json",
        {
            "rows": rows,
            "field_sha256": digest(DATA / "release-mixed-field-01.json"),
            "previous_inputs": PRIOR,
            "dolly_revision": DOLLY,
            "dolly_sha256": digest(path),
        },
    )
    held = [w.get("user_request", w["question"]) for w in rows]
    held_windows = [windows(q) for q in held]
    examples = []

    def add(row, group):
        question = user(row)
        q = windows(question)
        if any(q and h and len(q & h) / len(q | h) >= 0.30 for h in held_windows):
            return False
        examples.append({**row, "stratum": group})
        return True

    for w in original:
        add(w, "tools")
    for w in source["examples"]:
        add(
            w,
            {
                "MetaMathQA": "math",
                "BoolQ train": "judgment",
                "HotpotQA train": "reading",
            }[w["source"]],
        )
    source_file = Path("data/research_external/esnli_train_1.csv")
    n = Counter()
    with source_file.open(newline="") as f:
        for w in csv.DictReader(f):
            label = w["gold_label"]
            if label not in ("contradiction", "entailment") or n[label] >= 192:
                continue
            why = w["Explanation_1"].strip()
            if len(why.split()) < 4:
                continue
            q = f"Passage\n{w['Sentence1']}\n\nDoes the passage support the claim that {w['Sentence2']}\nAnswer yes or no."
            if norm(q) in queries:
                continue
            if add(
                {
                    "id": sha(q),
                    "source": "e-SNLI",
                    "messages": [{"role": "user", "content": q}],
                    "thinking": why,
                    "visible": "yes" if label == "entailment" else "no",
                },
                "judgment",
            ):
                n[label] += 1
                queries.add(norm(q))
            if sum(n.values()) == 384:
                break
    assert sum(n.values()) == 384
    targets = {"classification", "summarization", "information_extraction", "closed_qa"}
    n = Counter()
    for w in sorted(
        dolly, key=lambda r: sha("service-fit" + r["instruction"] + r["context"])
    ):
        cat = w["category"]
        ctx = w["context"].strip()
        inst = w["instruction"].strip()
        ans = w["response"].strip()
        if (
            cat not in targets
            or n[cat] >= 96
            or norm(ctx) in held_contexts
            or norm(inst) in queries
        ):
            continue
        q = (inst + "\n\n" + ctx).strip()
        if not (
            8 <= len(tok.encode(q, add_special_tokens=False)) <= 640
            and 1 <= len(tok.encode(ans, add_special_tokens=False)) <= 112
        ):
            continue
        if add(
            {
                "id": sha(q),
                "source": "Dolly " + cat,
                "messages": [{"role": "user", "content": q}],
                "thinking": "",
                "visible": ans,
            },
            "instruction",
        ):
            n[cat] += 1
            queries.add(norm(inst))
        if sum(n.values()) == 384:
            break
    assert len(n) == 4 and min(n.values()) == 96
    sources = {
        "reasoning": {
            "path": str(DATA / "shared-session-training.json"),
            "sha256": digest(DATA / "shared-session-training.json"),
        },
        "tools": {
            "path": str(DATA / "tool-dialogue-typed-training.json"),
            "sha256": digest(DATA / "tool-dialogue-typed-training.json"),
        },
        "e-SNLI": {"file": str(source_file), "sha256": digest(source_file)},
        "Dolly": {"revision": DOLLY, "sha256": digest(path)},
    }
    save(
        DATA / "unified-session-training.json",
        {"sources": sources, "examples": examples},
    )
    print(len(examples), Counter(w["stratum"] for w in examples), len(rows))


if __name__ == "__main__":
    main()
