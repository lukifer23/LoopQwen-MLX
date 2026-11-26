"""Create semantic tool-choice pairs from licensed demonstrations and executed recovery turns."""

import json
import random
import re
from pathlib import Path

from huggingface_hub import hf_hub_download
from jsonschema import Draft202012Validator
from program_reasoning_material import norm, sha
from tool_curriculum import HERMES, READ_ONLY, REV, TOPICS

from loopqwen.calculator_tool import calculate
from loopqwen.io import digest, read, save
from loopqwen.tool_session import SYSTEM, compact

DATA = Path("research/data")


def words(text):
    return set(norm(text.replace("_", " "))) - {
        "get",
        "find",
        "the",
        "for",
        "of",
        "a",
        "to",
        "or",
        "in",
        "and",
        "calculate",
        "given",
        "returns",
        "return",
        "at",
        "a",
        "can",
    }


def collection(excluded_names, excluded_questions):
    history = {}
    candidates = []
    for filename in ("func-calling-singleturn.json", "glaive-function-calling-5k.json"):
        path = Path(
            hf_hub_download(HERMES, filename, repo_type="dataset", revision=REV)
        )
        history[filename] = digest(path)
        for row in read(path):
            turns = row["conversations"]
            if [t["from"] for t in turns[:3]] != ["system", "human", "gpt"]:
                continue
            user = turns[1]["value"].strip()
            key = " ".join(norm(user))
            if key in excluded_questions or not 16 <= len(user) <= 380:
                continue
            match = re.fullmatch(
                r"\s*<tool_call>\s*(.*?)\s*</tool_call>\s*", turns[2]["value"], re.S
            )
            if not match:
                continue
            try:
                action = json.loads(match[1])
                name = action["name"]
                if (
                    name in excluded_names
                    or not READ_ONLY.search(name)
                    or not TOPICS.search(name)
                ):
                    continue
                fn = next(
                    d["function"]
                    for d in json.loads(row["tools"])
                    if d["function"]["name"] == name
                )
                params = fn["parameters"]
                if not isinstance(
                    action["arguments"], dict
                ) or not Draft202012Validator(params).is_valid(action["arguments"]):
                    continue
                if (
                    len(compact(fn)) > 1100
                    or len(compact(action)) > 470
                    or name in user
                ):
                    continue
            except (ValueError, TypeError, StopIteration, KeyError):
                continue
            excluded_questions.add(key)
            action = {"name": name, "arguments": action["arguments"]}
            candidates.append(
                (
                    filename + ":" + row["id"],
                    user,
                    fn,
                    action,
                    words(name + " " + fn["description"]),
                )
            )
    return history, sorted(candidates, key=lambda r: sha("intent-2031:" + r[0]))


def main():
    rng = random.Random(100391)
    panel = read(DATA / "paired-intent-panel.json")
    excluded = {" ".join(norm(r["user_request"])) for r in panel["rows"]}
    names = {n for r in panel["rows"] for n in r.get("names", [])}
    for row in panel["rows"]:
        names.update(f["name"] for f in row.get("function", []))
        if row["provider"] == "bfcl_exec":
            excluded.add(" ".join(norm(row["question"][0][0]["content"])))
    typed = read(DATA / "tool-dialogue-typed-training.json")["examples"]
    for name in (
        "tool-dialogue-typed-training",
        "tool-dialogue-grounded-training",
        "tool-counterfactual-training",
    ):
        for row in read(DATA / (name + ".json"))["examples"]:
            excluded.update(
                " ".join(norm(m["content"]))
                for m in row["messages"]
                if m["role"] == "user"
            )
    sources, bank = collection(names, excluded)
    # Each member asks its original independent question against the same two real
    # declarations. Favor neighboring meanings and distinct operations. Do not reuse a question.
    used = set()
    paired = []
    manifest = []
    for i, x in enumerate(bank):
        if i in used:
            continue
        peers = []
        for j, y in enumerate(bank):
            if j == i or j in used or x[2]["name"] == y[2]["name"]:
                continue
            overlap = x[4] & y[4]
            if not overlap:
                continue
            val = len(overlap) / len(x[4] | y[4])
            if val >= 0.09:
                peers.append((val, j))
        if not peers:
            continue
        _, j = max(peers)
        y = bank[j]
        used.update((i, j))
        n = len(manifest)
        visible_names = (
            [w[2]["name"] for w in (x, y)]
            if n % 2 == 0
            else [f"inspect_{rng.getrandbits(28):07x}" for _ in range(2)]
        )
        definitions = [
            dict(w[2], name=label)
            for w, label in zip((x, y), visible_names, strict=True)
        ]
        rng.shuffle(definitions)
        for order, w in enumerate((x, y)):
            action = dict(w[3], name=visible_names[order])
            messages = [
                dict(role="system", content=SYSTEM + compact(definitions)),
                dict(role="user", content=w[1]),
            ]
            paired.append(
                dict(
                    id=w[0],
                    pair=f"intent-{n}",
                    pair_order=order,
                    messages=messages,
                    thinking="",
                    visible=compact(action),
                )
            )
        manifest.append([x[0], y[0], x[2]["name"], y[2]["name"]])
        if len(manifest) == 256:
            break
    if len(manifest) < 128:
        raise ValueError(
            f"Only {len(manifest)} independently paired demonstrations from {len(bank)} eligible questions"
        )
    calc = read(DATA / "tool-program-reasoning-training.json")["examples"][:1280]
    by_id = {w["id"]: w for w in calc}
    original = [w for w in calc if w["id"].endswith("-call")]
    selected = rng.sample(original, 384)
    training = list(paired)
    for i, w in enumerate(selected):
        answer = by_id[w["id"].removesuffix("-call") + "-result"]
        action = json.loads(w["visible"])
        observation = calculate(**action["arguments"])
        assert observation == json.loads(answer["messages"][-1]["content"])
        training.extend([w, answer])
        if i < 192:
            # The runtime generates the error, and only the corrected program produces its observation.
            bad = dict(
                name="calculate",
                arguments={
                    "steps": [
                        s + " = " + v
                        for s, v in zip(
                            action["arguments"]["steps"],
                            observation["results"],
                            strict=True,
                        )
                    ]
                },
            )
            try:
                calculate(**bad["arguments"])
            except (ValueError, SyntaxError) as exc:
                error = {"error": type(exc).__name__}
            else:
                raise ValueError("The enrolled negative program actually executed")
            messages = [
                *w["messages"],
                dict(role="assistant", content=compact(bad)),
                dict(role="tool", content=compact(error)),
            ]
            training.append(dict(w, id=w["id"] + "-repair", messages=messages))
    sql = read(DATA / "tool-sql-learning-training.json")["examples"][:1536]
    for i in rng.sample(range(768), 192):
        training.extend(sql[2 * i : 2 * i + 2])
    training.extend(rng.sample(typed, 384))
    assert len({r["id"] for r in training}) == len(training)
    recipe = dict(
        repository=HERMES,
        revision=REV,
        source_files=sources,
        eligible=len(bank),
        pairs=manifest,
        distinct_names=len({n for p in manifest for n in p[2:]}),
        recovered_calculator_turns=192,
        executed_gsm_programs=384,
        wikisql_tables=192,
        typed_rehearsal=384,
        opaque_pairs=len(manifest) // 2,
    )
    save(DATA / "paired-intent-training.json", dict(examples=training, recipe=recipe))
    print(
        "training",
        len(training),
        "pairs",
        len(manifest),
        "source functions",
        recipe["distinct_names"],
        "eligible",
        len(bank),
    )


if __name__ == "__main__":
    main()
