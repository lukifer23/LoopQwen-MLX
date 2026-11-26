"""Freeze new follow-up requests against the actual installed Python standard library."""

from pathlib import Path

from mlx_lm.utils import load_tokenizer

from loopqwen.evaluation.module_questions import catalog, snapshot
from loopqwen.identity import fingerprint
from loopqwen.io import read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.tool_session import initial_messages, prompt


def freeze():
    spec = read("research/experiments/conversation-followup-modules-01.json")
    tok = load_tokenizer(local_snapshot())
    env = snapshot()
    tools = catalog()
    prefixes = sorted({n[0] for n in env}, key=lambda c: fingerprint(["stdlib", c]))
    groups = [(p, sorted(n for n in env if n.startswith(p))) for p in prefixes]
    groups = [g for g in groups if 2 <= len(g[1]) <= 10][:8]
    targets = {(g[-1] if j % 2 else g[0]) for j, (_, g) in enumerate(groups)}
    simple = sorted(
        set(env) - targets, key=lambda n: fingerprint(["clarify-source", n])
    )[:8]
    prompts = [
        "I need the number of lines in a Python standard-library module. What detail should I provide?",
        "I have a module in mind. Ask for its name before counting source lines.",
        "Can you check the source length for me? I forgot to say which module.",
        "I'd like a line count from this Python installation. Which module information do you need?",
        "Please help me measure a module's length in lines. Ask me for the missing name.",
        "Can we inspect how long a standard-library module is? I haven't specified the module yet.",
        "I want a source-line total. What should I tell you so you can look up the right module?",
        "I have an installed Python module to check. Ask which one, then find its line count.",
    ]
    rows = []
    for j, (prefix, names) in enumerate(groups):
        for kind in ("clarification", "reference"):
            ref = kind == "reference"
            target = (names[-1] if j % 2 else names[0]) if ref else simple[j]
            first = (
                f"List the installed standard-library modules whose names start with {prefix.upper() if j % 2 else prefix}. Use the local inventory."
                if ref
                else prompts[j]
            )
            follow = (
                f"How many lines of source are in the alphabetically {'last' if j % 2 else 'first'} one?"
                if ref
                else target + "."
            )
            rows.append(
                {
                    "id": kind + "-" + target,
                    "kind": kind,
                    "resource": "module",
                    "target": target,
                    "lines": env[target]["lines"],
                    "first": first,
                    "reply": follow,
                    "prefix": prefix if ref else None,
                    "matches": names if ref else [],
                    "initial_ids": tok.encode(
                        prompt(tok, initial_messages(first, tools)),
                        add_special_tokens=False,
                    ),
                }
            )
    assert len({w["target"] for w in rows}) == 16
    mat = {"environment": env, "rows": rows}
    path = Path(spec["material"])
    if path.is_file() and read(path) != mat:
        raise ValueError("Frozen interpreter source differs")
    save(path, mat)


if __name__ == "__main__":
    freeze()
