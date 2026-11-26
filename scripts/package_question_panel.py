"""Freeze multi-step questions about the actual installed Python environment."""

import hashlib

from loopqwen.io import save
from loopqwen.package_tools import snapshot

if __name__ == "__main__":
    environment = snapshot()
    order = sorted(
        environment,
        key=lambda s: hashlib.sha256(("local-packages-476:" + s).encode()).digest(),
    )
    rows = []
    target_counts = {}
    for name in order:
        deps = environment[name]["dependencies"]
        if not 2 <= len(deps) <= 12 or not all(d in environment for d in deps):
            continue
        first = len(rows) % 2 == 0
        target = deps[0 if first else -1]
        target_counts[target] = target_counts.get(target, 0) + 1
        position = "first" if first else "last"
        question = f"For the installed Python distribution {name}, which version is installed of its alphabetically {position} active direct dependency? Check the dependencies and then the version using the tools. Give the version number."
        rows.append(
            dict(
                id="local-deps-" + name,
                root=name,
                target=target,
                dependencies=deps,
                position=position,
                version=environment[target]["version"],
                user_request=question,
            )
        )
    if not rows:
        raise ValueError(
            "No installed distributions have a resolvable dependency choice"
        )
    save(
        "research/data/tool-package-dependencies-01.json",
        dict(environment=environment, rows=rows),
    )
    print(
        len(rows), "installed roots,", len(target_counts), "distinct dependency targets"
    )
