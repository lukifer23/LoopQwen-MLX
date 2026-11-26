"""Freeze a mixed practical stream and the separate reasoning and routing supervision."""

import json
import random
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from contrast_material import REV as BOOL_REV
from contrast_material import key
from huggingface_hub import hf_hub_download
from mlx_lm.utils import load_tokenizer
from operand_material import external
from program_reasoning_material import former, norm, sha

from loopqwen.calculator_tool import tool as calculator
from loopqwen.evaluation import executable_tools
from loopqwen.evaluation.answers import numeric
from loopqwen.evaluation.mixed_field import catalog
from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.tool_session import initial_messages, prompt, system_message

DATA = Path("research/data")
PRIOR = "bfe01c6"
SMOL = "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa"


def parquet(repo, file, revision):
    return Path(hf_hub_download(repo, file, repo_type="dataset", revision=revision))


def opening(text):
    return tuple(norm(text)[:8])


def panel(tok, field):
    external_math = external()
    path = parquet("google/boolq", "data/validation-00000-of-00001.parquet", BOOL_REV)
    labels = pq.read_table(path).to_pylist()
    excluded = former(
        {q for _, q, _ in external_math} | {w["passage"] for w in labels}, prior=PRIOR
    )
    blocked = {opening(q) for q in excluded}
    rows = []
    counts = Counter()
    for source, question, gold in sorted(
        external_math, key=lambda r: sha("shared-new-math" + r[1])
    ):
        if gold is None or counts[source] == 12 or opening(question) in blocked:
            continue
        rows.append(
            {
                "family": "math",
                "source": source,
                "id": sha(question),
                "question": question + "\nGive the final numeric answer.",
                "gold": gold,
            }
        )
        counts[source] += 1
        blocked.add(opening(question))
    for r in sorted(
        labels, key=lambda r: sha("shared-new-reading" + r["passage"] + r["question"])
    ):
        if counts[r["answer"]] == 12 or opening(r["passage"]) in blocked:
            continue
        user = f"Passage\n{r['passage']}\n\nQuestion\n{r['question']}\nAnswer yes or no based on the passage."
        rows.append(
            {
                "family": "judgment",
                "source": "BoolQ validation",
                "id": key(r["passage"]),
                "question": user,
                "gold": "yes" if r["answer"] else "no",
            }
        )
        counts[r["answer"]] += 1
        blocked.add(opening(r["passage"]))
    assert sorted(counts.values()) == [12] * 4
    rows.extend({**w, "source": "field retention"} for w in field["rows"])
    for w in read(DATA / "tool-exec-bfcl-01.json")["rows"]:
        user = w["question"][0][0]["content"] + executable_tools.INSTRUCTION
        rows.append(
            {
                **w,
                "family": "public",
                "source": "public retention",
                "user_request": user,
            }
        )
    random.Random(81943).shuffle(rows)
    for w in rows:
        family = w["family"]
        tools = (
            []
            if family in ("math", "judgment")
            else executable_tools.tools_for(w)
            if family == "public"
            else catalog(w, field, True)
        )
        user = w.get("user_request", w["question"])
        messages = (
            initial_messages(user, tools)
            if tools
            else [{"role": "user", "content": user}]
        )
        ids = tok.encode(prompt(tok, messages), add_special_tokens=False)
        assert w.get("initial_ids", ids) == ids
        w["initial_ids"] = ids
    assert len(rows) == len({w["id"] for w in rows}) == 120
    save(
        DATA / "shared-session-panel.json",
        {
            "rows": rows,
            "field_sha256": digest(DATA / "release-mixed-field-01.json"),
            "previous_inputs": PRIOR,
        },
    )
    return rows


def cleaned(text):
    return re.findall(
        r"[a-z]+|<n>", re.sub(r"[0-9]+(?:\.[0-9]+)?", "<n>", text.lower())
    )


def windows(text):
    words = cleaned(text)
    return {tuple(words[i : i + 5]) for i in range(len(words) - 4)}


def main():
    tok = load_tokenizer(local_snapshot())
    field = read(DATA / "release-mixed-field-01.json")
    held = panel(tok, field)
    held_text = [w.get("user_request", w["question"]) for w in held]
    held_windows = [windows(q) for q in held_text]
    seen = set()
    sources = {}
    examples = []
    routing = []
    rng = random.Random(62911)
    calc = system_message([calculator()])["content"]
    logged = read(DATA / "dialogue-state-training.json")["examples"]
    logged = [w for w in logged if w.get("kind") in ("ask", "call")]
    service_schemas = sorted({w["messages"][0]["content"] for w in logged})

    def enroll(source, question, thinking, answer, family):
        identifier = sha(question)
        if identifier in seen:
            return False
        gs = windows(question)
        if any(gs and h and len(gs & h) / len(gs | h) >= 0.30 for h in held_windows):
            return False
        seen.add(identifier)
        messages = [{"role": "user", "content": question}]
        # Independent examples teach a direct reply both with and without available tools.
        schema = (
            calc
            if family == "math" and rng.random() < 0.62
            else rng.choice(service_schemas)
        )
        if rng.random() < 0.46:
            messages.insert(0, {"role": "system", "content": schema})
        examples.append(
            {
                "id": identifier,
                "source": source,
                "messages": messages,
                "thinking": thinking,
                "visible": answer,
            }
        )
        routing.append(
            {
                "id": identifier,
                "question": question,
                "schema": schema,
                "external": False,
            }
        )
        return True

    meta = parquet(
        "HuggingFaceTB/smoltalk",
        "data/metamathqa-50k/train-00000-of-00001.parquet",
        SMOL,
    )
    sources["MetaMathQA"] = {
        "repo": "HuggingFaceTB/smoltalk",
        "revision": SMOL,
        "file": meta.name,
        "sha256": digest(meta),
    }
    mathrows = pq.read_table(meta).to_pylist()
    count = 0
    for w in sorted(
        mathrows, key=lambda w: sha("shared-math" + w["messages"][0]["content"])
    ):
        msgs = w["messages"]
        if len(msgs) != 2:
            continue
        question = msgs[0]["content"].strip()
        work, sep, target = msgs[1]["content"].rpartition("The answer is:")
        gold = numeric(target.strip().rstrip("."))
        if (
            not sep
            or gold is None
            or not 16 <= len(tok.encode(work, add_special_tokens=False)) <= 112
            or len(question) > 650
        ):
            continue
        if enroll("MetaMathQA", question, work.strip(), gold, "math"):
            count += 1
        if count == 1536:
            break
    assert count == 1536
    train = parquet("google/boolq", "data/train-00000-of-00001.parquet", BOOL_REV)
    sources["BoolQ"] = {
        "repo": "google/boolq",
        "revision": BOOL_REV,
        "file": train.name,
        "sha256": digest(train),
    }
    counts = Counter()
    for w in sorted(
        pq.read_table(train).to_pylist(),
        key=lambda w: sha("shared-read" + w["passage"] + w["question"]),
    ):
        if counts[w["answer"]] == 384 or len(w["passage"]) > 1600:
            continue
        user = f"Passage\n{w['passage']}\n\nQuestion\n{w['question']}\nAnswer yes or no based on the passage."
        if enroll("BoolQ train", user, "", "yes" if w["answer"] else "no", "text"):
            counts[w["answer"]] += 1
        if sum(counts.values()) == 768:
            break
    assert sorted(counts.values()) == [384, 384]
    hot = DATA / "linked-reading-train.json"
    sources["HotpotQA"] = {"material": str(hot), "sha256": digest(hot)}
    n = 0
    for w in sorted(read(hot)["rows"], key=lambda w: sha("shared-hotpot" + w["id"])):
        passages = "\n\n".join(
            t + "\n" + a for t, a in zip(w["titles"], w["articles"], strict=True)
        )
        user = (
            passages
            + "\n\n"
            + w["question"]
            + "\nGive a short answer from the passages."
        )
        if len(tok.encode(user, add_special_tokens=False)) > 1540:
            continue
        evidence = " and ".join(w["titles"][i] for i in w["support"]) + "."
        if enroll("HotpotQA train", user, evidence, w["gold"], "text"):
            n += 1
        if n == 256:
            break
    assert n == 256
    # Positive labels use external requests only. No sampled API definition is executed.
    for w in logged:
        q = next(m["content"] for m in reversed(w["messages"]) if m["role"] == "user")
        routing.append(
            {
                "id": "sgd-" + w["id"],
                "question": q,
                "schema": w["messages"][0]["content"],
                "external": True,
            }
        )
    apis = parquet(
        "HuggingFaceTB/smoltalk", "data/apigen-80k/train-00000-of-00001.parquet", SMOL
    )
    sources["APIGen"] = {
        "repo": "HuggingFaceTB/smoltalk",
        "revision": SMOL,
        "file": apis.name,
        "sha256": digest(apis),
    }
    external_topics = re.compile(
        r"weather|forecast|books?|movies?|flights?|hotel|restaurant|music|recipe|train.schedule|museum|stock.price|exchange.rate",
        re.IGNORECASE,
    )
    safe_name = re.compile(
        r"(?:get|find|look|check|read|search|list|fetch|retrieve|show|view|query|forecast|current|convert)",
        re.IGNORECASE,
    )
    seen_api = set()
    for batch in pq.ParquetFile(apis).iter_batches(batch_size=512):
        for w in batch.to_pylist():
            msgs = w["messages"]
            if len(msgs) != 3 or [m["role"] for m in msgs] != [
                "system",
                "user",
                "assistant",
            ]:
                continue
            block = re.search(r"<tools>(.*?)</tools>", msgs[0]["content"], re.DOTALL)
            calls = re.search(r"<tool_call>(.*?)</tool_call>", msgs[2]["content"], re.DOTALL)
            if not block or not calls:
                continue
            try:
                defs = json.loads(block[1])
                actions = json.loads(calls[1])
            except ValueError:
                continue
            if (
                not isinstance(actions, list)
                or len(actions) != 1
                or not isinstance(actions[0], dict)
            ):
                continue
            name = actions[0].get("name", "")
            if not safe_name.match(name) or not external_topics.search(name):
                continue
            schema = next((f for f in defs if f.get("name") == name), None)
            user = msgs[1]["content"]
            if (
                not schema
                or len(user) > 500
                or sha(user) in seen_api
                or not external_topics.search(user)
            ):
                continue
            desc = schema.get("description", "")
            if len(desc) > 500:
                continue
            seen_api.add(sha(user))
            routing.append(
                {
                    "id": "apigen-" + sha(user),
                    "question": user,
                    "schema": json.dumps(
                        schema, ensure_ascii=False, separators=(",", ":")
                    ),
                    "external": True,
                }
            )
            if len(seen_api) == 768:
                break
        if len(seen_api) == 768:
            break
    print("Eligible independent APIGen requests:", len(seen_api), flush=True)
    assert len(seen_api) >= 128
    typed = read(DATA / "tool-dialogue-typed-training.json")["examples"]
    typedq = set()
    for w in typed:
        msgs = w["messages"]
        if len(msgs) != 2 or not w["visible"].startswith('{"name":'):
            continue
        user = msgs[-1]["content"]
        if sha(user) in typedq:
            continue
        typedq.add(sha(user))
        routing.append(
            {
                "id": "typed-" + sha(user),
                "question": user,
                "schema": msgs[0]["content"],
                "external": True,
            }
        )
    # Group by normalized question so the same request cannot enter both router splits.
    for w in routing:
        group = sha("shared-router-split" + " ".join(norm(w["question"])))
        w["split"] = "validation" if int(group[:8], 16) % 9 == 0 else "train"
    save(
        DATA / "shared-session-training.json",
        {"sources": sources, "examples": examples, "router": routing},
    )
    print(
        len(examples), Counter((w["external"], w["split"]) for w in routing), flush=True
    )


if __name__ == "__main__":
    main()
