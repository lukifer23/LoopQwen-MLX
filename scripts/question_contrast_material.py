"""Freeze human question changes on common source text and an untouched DROP comparison."""

import ast
import itertools
import json
import re
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile

from mlx_lm.utils import load_tokenizer
from operand_material import external, grams
from program_reasoning_material import former, norm, sha, source

from loopqwen import qwen_tools
from loopqwen.calculator_tool import tool
from loopqwen.evaluation.answers import numeric
from loopqwen.evaluation.program_questions import INSTRUCTION
from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.research.native_function_data import catalog
from loopqwen.tool_session import compact

DATA = Path("research/data")
PRIOR = "086528e"
SRC = Path("data/research_external")
DROP_SHA = "39d2278a29fd729de301b111a45f434c24834f40df8f4ff116d864589e3249d6"
SQUAD_SHA = "3527663986b8295af4f7fcdff1ba1ff3f72d07d61a20f487cb238a6ef92fd955"
CALC = tool()
TOOLS = qwen_tools.declarations([CALC])
NUMBER = re.compile(r"\d+(?:\.\d+)?")
READING = "\nGive the numerical answer. You may use the calculator for arithmetic."


def fresh_numeric(q):
    result = [
        numeric(q["answer"]["number"]),
        *[numeric(a["number"]) for a in q.get("validated_answers", [])],
    ]
    return (
        result[0]
        if len(result) >= 2 and result[0] is not None and len(set(result)) == 1
        else None
    )


def choose_pair(items, salt):
    for a, b in itertools.combinations(
        sorted(items, key=lambda w: sha(salt + w[0])), 2
    ):
        if a[1] != b[1] and a[0] != b[0]:
            return [a, b]


def example(ident, user, target, messages=()):
    return {
        "id": ident,
        "tools": TOOLS,
        "messages": [{"role": "user", "content": user}, *messages],
        "target": target,
    }


def answer(v):
    return {"role": "assistant", "content": str(v)}


def action(steps):
    return qwen_tools.assistant_message(
        [{"name": "calculate", "arguments": {"steps": steps}}]
    )


def call_and_return(steps):
    return [action(steps), {"role": "tool", "content": compact(CALC.callback(steps))}]


def number_map(first, second):
    a, b = (NUMBER.findall(w["Body"]) for w in (first, second))
    if len(a) != len(b) or len(set(b)) != len(b) or len(set(a)) != len(a):
        return None
    mapping = dict(zip(b, a, strict=True))
    if NUMBER.sub(lambda m: mapping[m[0]], second["Body"]) != first["Body"]:
        return None
    return mapping


def moved_equation(w, mapping):
    tree = ast.parse(w["Equation"], mode="eval")
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant):
            key = str(Fraction(str(n.value)))
            if key in mapping:
                n.value = (
                    int(mapping[key])
                    if "." not in mapping[key]
                    else float(mapping[key])
                )
    return ast.unparse(tree.body)


def main():
    tok = load_tokenizer(local_snapshot())
    drop = source(
        "drop_dataset.zip",
        "https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip",
        DROP_SHA,
    )
    squad = source(
        "squad-train-v1.1.json",
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
        SQUAD_SHA,
    )
    with ZipFile(drop) as z:
        dev = json.loads(z.read("drop_dataset/drop_dataset_dev.json"))
    field = []
    docs = []
    for pid, w in sorted(dev.items(), key=lambda x: sha("human-drop-pair-205" + x[0])):
        vals = [
            (q["question"], v, q["query_id"])
            for q in w["qa_pairs"]
            if (v := fresh_numeric(q)) is not None
        ]
        pair = choose_pair(vals, "drop-choice-206")
        if not pair:
            continue
        for q, v, k in pair:
            field.append(
                {
                    "id": k,
                    "cluster": pid,
                    "family": "DROP",
                    "gold": v,
                    "questions": [w["passage"] + "\n\n" + q + READING],
                }
            )
        docs.append(w["passage"])
        if len(docs) == 24:
            break
    original = external()
    pool = [(f, q, v) for f, q, v in original if f == "ASDiv"]
    svamp = read(SRC / "svamp.json")

    def full(w):
        return w["Body"].strip() + " " + w["Question"].strip()

    groups = []
    for article in read(squad)["data"]:
        for par in article["paragraphs"]:
            vals = []
            for q in par["qas"]:
                if len(q["answers"]) == 1:
                    v = numeric(q["answers"][0]["text"])
                    if v is not None:
                        vals.append((q["question"], v, q["id"]))
            if len({v[0] for _, v, _ in vals}) >= 2:
                groups.append((par["context"], vals))
    texts = {q for _, q, _ in pool} | {full(w) for w in svamp} | {s for s, _ in groups}
    blocked = {tuple(norm(q)[:8]) for q in former(texts, prior=PRIOR)}
    fresh = 0
    for f, q, v in sorted(pool, key=lambda w: sha("human-target-207" + w[1])):
        k = tuple(norm(q)[:8])
        if v is None or k in blocked:
            continue
        field.append(
            {
                "id": sha(q),
                "cluster": sha(q),
                "family": f,
                "questions": [q + INSTRUCTION],
                "gold": v,
            }
        )
        blocked.add(k)
        fresh += 1
        if fresh == 24:
            break
    assert fresh == 24, fresh
    held = [grams(r["questions"][0]) for r in field]

    def distinct(s):
        g = grams(s)
        return not any(len(g & h) / max(1, len(g | h)) >= 0.4 for h in held)

    # Exclude exact text excerpts shared across the two distinct reading corpora.
    def shingles(s):
        words = norm(s)
        return {tuple(words[i : i + 12]) for i in range(len(words) - 11)}

    forbidden = set().union(*(shingles(s) for s in docs))
    rows = []
    manifest = []
    for context, vals in sorted(groups, key=lambda v: sha("human-squad-208" + v[0])):
        if tuple(norm(context)[:8]) in blocked or shingles(context) & forbidden:
            continue
        pair = next(
            (
                p
                for p in itertools.combinations(
                    sorted(vals, key=lambda v: sha(v[2])), 2
                )
                if p[0][1][0] != p[1][1][0]
            ),
            None,
        )
        if not pair:
            continue
        group = sha(context)
        for i, (q, v, k) in enumerate(pair):
            user = context + "\n\n" + q + READING
            rows.append(
                dict(example(k, user, answer(v)), pair=group, pair_order=i, thinking="")
            )
        manifest.append({"kind": "SQuAD", "id": group, "rows": [p[2] for p in pair]})
        if len(manifest) == 256:
            break
    assert len(manifest) == 256
    bodies = defaultdict(list)
    for w in svamp:
        if tuple(norm(full(w))[:8]) not in blocked and distinct(full(w)):
            bodies[NUMBER.sub("@", w["Body"])].append(w)
    math_pairs = 0
    for story, group in sorted(
        bodies.items(), key=lambda x: sha("human-math-209" + x[0])
    ):
        for a, b in itertools.combinations(sorted(group, key=lambda x: x["ID"]), 2):
            mapping = number_map(a, b)
            if mapping is None or a["Question"] == b["Question"]:
                continue
            if NUMBER.search(a["Question"]) or NUMBER.search(b["Question"]):
                continue
            if any(
                key != value and min(Fraction(key), Fraction(value)) <= 2
                for key, value in mapping.items()
            ):
                continue
            if any(
                Fraction(CALC.callback([w["Equation"]])["value"])
                != Fraction(str(w["Answer"]))
                for w in (a, b)
            ):
                continue
            try:
                expressions = [moved_equation(a, {}), moved_equation(b, mapping)]
                values = [CALC.callback([e])["value"] for e in expressions]
            except (ValueError, ZeroDivisionError):
                continue
            if values[0] == values[1]:
                continue
            questions = [
                full(a),
                a["Body"].strip() + " " + b["Question"].strip(),
            ]
            if not all(distinct(q) for q in questions):
                continue
            ident = a["ID"] + "-" + b["ID"]
            for i, q in enumerate(questions):
                user = q + INSTRUCTION
                right = call_and_return([expressions[i]])
                wrong = call_and_return([expressions[1 - i]])
                rows.extend(
                    [
                        dict(
                            example(ident + f"-{i}-action", user, right[0]),
                            pair=ident,
                            pair_order=i,
                            thinking="",
                        ),
                        example(ident + f"-{i}-accept", user, answer(values[i]), right),
                        example(ident + f"-{i}-revise", user, right[0], wrong),
                        example(
                            ident + f"-{i}-finish",
                            user,
                            answer(values[i]),
                            [*wrong, *right],
                        ),
                    ]
                )
            manifest.append(
                {
                    "kind": "SVAMP",
                    "id": ident,
                    "rows": [a["ID"], b["ID"]],
                    "map": mapping,
                    "questions": questions,
                    "programs": expressions,
                    "returns": values,
                }
            )
            math_pairs += 1
            break
    assert math_pairs >= 12, math_pairs
    native = DATA / "native-function-training.json"
    rows.extend(
        sorted(
            (
                v
                for v in read(native)["examples"]
                if distinct(v["messages"][0]["content"])
            ),
            key=lambda v: sha("human-rehearsal-210" + v["id"]),
        )[:640]
    )
    original_panel = read(DATA / "native-function-panel.json")
    for w in original_panel["rows"]:
        if w["family"] in ("public", "packages", "files"):
            field.append(dict(w, cluster=w["id"]))
    field.sort(key=lambda w: sha("human-contrast-order-211" + w["id"]))
    for w in field:
        tools = (
            TOOLS
            if w["family"] in ("DROP", "ASDiv")
            else qwen_tools.declarations(catalog(w))
        )
        ids = tok.encode(
            qwen_tools.render(
                tok, [{"role": "user", "content": w["questions"][0]}], tools
            ),
            add_special_tokens=False,
        )
        assert w.get("initial_ids", ids) == ids
        w["initial_ids"] = ids
    for name, d in [
        (
            "training",
            {
                "examples": rows,
                "pairs": manifest,
                "sources": {
                    str(p): digest(p) for p in [squad, SRC / "svamp.json", native]
                },
            },
        ),
        (
            "panel",
            {
                "rows": field,
                "sources": {str(p): digest(p) for p in [drop, SRC / "asdiv.xml"]},
                "environment": original_panel["environment"],
                "files": original_panel["files"],
            },
        ),
    ]:
        dest = DATA / f"question-contrast-{name}.json"
        if dest.exists():
            assert read(dest) == d
        else:
            save(dest, d)
    print(
        "SQuAD", 256, "SVAMP", math_pairs, "targets", len(rows), "dialogues", len(field)
    )


if __name__ == "__main__":
    main()
