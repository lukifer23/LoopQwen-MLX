"""Freeze two-user-turn requests using a disjoint set of installed distributions."""

from pathlib import Path

from mlx_lm.utils import load_tokenizer

from loopqwen.evaluation.conversation_questions import catalog
from loopqwen.identity import fingerprint
from loopqwen.io import read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.package_tools import snapshot
from loopqwen.tool_session import initial_messages, prompt


def freeze():
    spec = read("research/experiments/conversation-followup-01.json")
    environment = snapshot()
    chains = read("research/data/tool-package-dependencies-01.json")
    mixed = read(read("research/experiments/release-mixed-field-01.json")["material"])
    earlier = read("research/experiments/tool-language-release-01.json")
    used = {
        "loopqwen",
        *(n.lower() for n in earlier["packages"]),
        *(w[k] for w in chains["rows"] for k in ("root", "target")),
        *(w["name"] for w in mixed["rows"] if w["family"] == "package"),
    }
    tok = load_tokenizer(local_snapshot())
    options = sorted(
        set(environment) - used, key=lambda name: fingerprint(["dialogue", name])
    )
    references = []
    for prefix in sorted({k[0] for k in environment}):
        names = sorted(k for k in environment if k.startswith(prefix))
        if len(names) < 2:
            continue
        choices = [
            (place, name)
            for place, name in (("first", names[0]), ("last", names[-1]))
            if name not in used
        ]
        if choices:
            place, name = choices[len(references) % len(choices)]
            references.append((place, name, prefix, names))
    references = references[:8]
    if len(references) != 8:
        raise ValueError("Need eight distinct live prefixes with unused end entries")
    simple = [n for n in options if n not in {r[1] for r in references}][:8]
    starts = [
        "I need a package version from this environment. What do you need to know?",
        "Can you check an installed version for me? Ask for the package name first.",
        "I forgot to name the package whose version I need. Ask me for it.",
        "Help me look up a distribution version. What information is missing?",
        "I want the installed version of one distribution. Which detail do you need?",
        "Can you find a library version on this machine? I haven't told you which library.",
        "Please ask which package to inspect before checking the installed version.",
        "I have a question about a package version. What should I tell you first?",
    ]
    rows = []
    for j in range(8):
        order, target, prefix, names = references[j]
        for kind, name in (("clarification", simple[j]), ("reference", target)):
            first = (
                starts[j]
                if kind == "clarification"
                else (
                    f"Which installed Python distributions start with the letter {prefix.upper()}? Check this environment."
                )
            )
            second = (
                name + "."
                if kind == "clarification"
                else (
                    f"What installed version does the alphabetically {order} one have?"
                )
            )
            rows.append(
                {
                    "id": kind + "-" + name,
                    "kind": kind,
                    "target": name,
                    "version": environment[name]["version"],
                    "prefix": prefix if kind == "reference" else None,
                    "matches": names if kind == "reference" else [],
                    "first": first,
                    "reply": second,
                    "initial_ids": tok.encode(
                        prompt(tok, initial_messages(first, catalog())),
                        add_special_tokens=False,
                    ),
                }
            )
    if len({r["target"] for r in rows}) != 16:
        raise ValueError("Need sixteen different held-out final lookup targets")
    material = {"environment": environment, "rows": rows, "excluded": sorted(used)}
    path = Path(spec["material"])
    if path.exists() and read(path) != material:
        raise ValueError("Frozen package dialogues differ")
    save(path, material)


if __name__ == "__main__":
    freeze()
