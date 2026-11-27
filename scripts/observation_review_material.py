"""Freeze paired real tool returns and new publisher arithmetic before fitting."""

import ast
import copy
import itertools
from collections import Counter
from fractions import Fraction
from pathlib import Path

from mlx_lm.utils import load_tokenizer
from operand_material import external, grams
from program_reasoning_material import former, norm, sha, source

from loopqwen import package_tools, qwen_tools
from loopqwen.calculator_tool import tool
from loopqwen.evaluation.program_questions import INSTRUCTION
from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.research.native_function_data import catalog, from_compact
from loopqwen.tool_session import compact

DATA = Path("research/data")
PRIOR = "4c4ecf2"
CALC = tool()


def alternatives(steps):
    # Change the meaning while preserving valid references and the original numbers.
    if len(steps) > 1:
        yield steps[:-1]
    replacements = {
        ast.Add: ast.Sub,
        ast.Sub: ast.Add,
        ast.Mult: ast.Div,
        ast.Div: ast.Mult,
    }
    for i in range(len(steps) - 1, -1, -1):
        tree = ast.parse(steps[i], mode="eval")
        for index, n in enumerate(ast.walk(tree)):
            if not isinstance(n, ast.BinOp) or type(n.op) not in replacements:
                continue
            for swap in (False, True):
                altered = copy.deepcopy(tree)
                node = list(ast.walk(altered))[index]
                if swap:
                    node.left, node.right = node.right, node.left
                else:
                    node.op = replacements[type(n.op)]()
                yield steps[:i] + [ast.unparse(altered.body)] + steps[i + 1 :]


def task(row):
    steps = row["steps"]
    gold = CALC.callback(steps)
    assert Fraction(gold["value"]) == Fraction(row["gold"])
    for altered in sorted(alternatives(steps), key=lambda x: sha(row["id"] + str(x))):
        try:
            wrong = CALC.callback(altered)
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            continue
        if Fraction(wrong["value"]) == Fraction(gold["value"]):
            continue

        def call(values):
            return {"name": "calculate", "arguments": {"steps": values}}

        return {
            "id": row["id"],
            "source": row["source"],
            "question": row["question"] + INSTRUCTION,
            "right": call(steps),
            "wrong": call(altered),
            "reference": gold["value"],
        }


def turn(c, available):
    observation = available[c["name"]].callback(**c["arguments"])
    return [
        qwen_tools.assistant_message([c]),
        {"role": "tool", "content": compact(observation)},
    ]


def lessons(row, tools):
    available = {t.name: t for t in tools}
    user = {"role": "user", "content": row["question"]}
    good, bad = (turn(row[k], available) for k in ("right", "wrong"))
    assert good[1]["content"] != bad[1]["content"]
    answer = {"role": "assistant", "content": row["reference"]}
    for kind, history, target in [
        ("accept", [user, *good], answer),
        ("revise", [user, *bad], good[0]),
        ("finish", [user, *bad, *good], answer),
    ]:
        yield {
            "id": row["id"] + "-" + kind,
            "tools": qwen_tools.declarations(tools),
            "messages": history,
            "target": target,
        }


def main():
    tok = load_tokenizer(local_snapshot())
    external_rows = external()
    old = former({q for _, q, _ in external_rows}, prior=PRIOR)
    blocked = {tuple(norm(q)[:8]) for q in old}
    primary = []
    counts = Counter()
    for fam, q, gold in sorted(
        external_rows, key=lambda r: sha("review-field-1402" + r[1])
    ):
        key = tuple(norm(q)[:8])
        if gold is None or key in blocked or counts[fam] >= 24:
            continue
        blocked.add(key)
        counts[fam] += 1
        primary.append(
            {"id": sha(q), "family": fam, "questions": [q + INSTRUCTION], "gold": gold}
        )
    assert sorted(counts.values()) == [24, 24]
    human = list(primary)
    prev = read(DATA / "native-function-panel.json")
    primary.extend(
        r for r in prev["rows"] if r["family"] in ("public", "files", "packages")
    )
    primary.sort(key=lambda r: sha("observation-order-1404" + r["id"]))
    for w in primary:
        text = qwen_tools.render(
            tok,
            [{"role": "user", "content": w["questions"][0]}],
            qwen_tools.declarations(catalog(w)),
        )
        ids = tok.encode(text, add_special_tokens=False)
        assert w.get("initial_ids", ids) == ids
        w["initial_ids"] = ids
    src = read("research/experiments/checked-rationale-01.json")["sources"]["svamp"]
    path = source(src["file"], src["url"], src["sha256"])
    probe = []
    for w in sorted(read(path), key=lambda r: sha("observation-check-1405" + r["ID"])):
        q = w["Body"].strip() + " " + w["Question"].strip()
        k = tuple(norm(q)[:8])
        if k in blocked:
            continue
        row = {
            "source": "SVAMP",
            "id": sha(q),
            "question": q,
            "steps": [w["Equation"]],
            "gold": str(w["Answer"]),
        }
        ex = task(row)
        if ex:
            probe.append(ex)
            blocked.add(k)
        if len(probe) == 24:
            break
    assert len(probe) == 24
    held = [grams(w["questions"][0].removesuffix(INSTRUCTION)) for w in human] + [
        grams(w["question"].removesuffix(INSTRUCTION)) for w in probe
    ]
    source_path = DATA / "draft-program-training.json"
    pool = read(source_path)["rows"]
    math = []
    counts = Counter()
    for w in sorted(pool, key=lambda r: sha("review-fit-1406" + r["id"])):
        g = grams(w["question"])
        if counts[w["source"]] >= 160 or any(
            len(g & b) / max(len(g | b), 1) >= 0.4 for b in held
        ):
            continue
        ex = task(w)
        if ex:
            math.append(ex)
            counts[w["source"]] += 1
    assert sorted(counts.values()) == [160, 160]
    examples = [e for w in math for e in lessons(w, [CALC])]
    env = package_tools.snapshot()
    assert env == prev["environment"]
    held_names = {
        c["arguments"]["distribution"]
        for w in primary
        if w["family"] == "packages"
        for c in w["calls"]
    }
    names = sorted(set(env) - held_names, key=lambda n: sha("package-review-1407" + n))
    for i, name in enumerate(names):
        other = next(
            n
            for n in itertools.islice(itertools.cycle(names), i + 1, i + 1 + len(names))
            if env[n]["version"] != env[name]["version"]
        )
        call = lambda n: {"name": "package_version", "arguments": {"distribution": n}}
        w = {
            "id": "env-" + name,
            "question": f"Read the installed version of {name}.",
            "right": call(name),
            "wrong": call(other),
            "reference": env[name]["version"],
        }
        examples.extend(lessons(w, package_tools.tools()))
    typed = DATA / "tool-dialogue-typed-training.json"
    archive = [
        r
        for r in read(typed)["examples"]
        if r["messages"][-1]["role"] == "tool" and not r["visible"].startswith("{")
    ]
    for r in sorted(archive, key=lambda r: sha("review-typed-1410" + r["id"]))[:128]:
        examples.append(from_compact(r))
    panel = {
        "rows": primary,
        "probe": probe,
        "files": prev["files"],
        "environment": env,
        "excluded_through": PRIOR,
        "source_sha256": src["sha256"],
    }
    training = {
        "examples": examples,
        "paired_math": math,
        "sources": {str(p): digest(p) for p in (source_path, typed)},
        "package_pairs": len(names),
    }
    for name, content in [
        ("observation-review-panel", panel),
        ("observation-review-training", training),
    ]:
        dest = DATA / (name + ".json")
        if dest.exists():
            assert read(dest) == content
        else:
            save(dest, content)
    print(len(primary), len(probe), len(math), len(examples))


if __name__ == "__main__":
    main()
