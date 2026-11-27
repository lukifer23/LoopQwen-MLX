"""Enroll fresh arithmetic and resource conversations and adapt publisher data to official Qwen messages."""

import subprocess
from collections import Counter
from pathlib import Path
from mlx_lm.utils import load_tokenizer
from operand_material import external, grams
from program_reasoning_material import former, norm, sha
from loopqwen import package_tools, qwen_tools
from loopqwen.io import read, save, digest
from loopqwen.model_loader import local_snapshot
from loopqwen.research.native_function_data import (
    catalog,
    from_compact,
    service_targets,
    token_pair,
)
from loopqwen.evaluation.program_questions import INSTRUCTION

DATA = Path("research/data")
PRIOR = "be3cc54"
# These held-out resource requests use fresh package/file combinations. Each
# conversation changes the answer and alternate wording is reserved for testing.
PAIRS = [
    (
        "Can you look up the installed version of {}?",
        "And what version of {} do I have here?",
    ),
    ("¿Qué versión de {} hay instalada?", "¿Y la versión de {} en este entorno?"),
    (
        "Quelle version de {} est installée ici ?",
        "Et pour {}, quelle est la version installée ?",
    ),
    (
        "Welche Version von {} ist hier installiert?",
        "Und welche Version von {} habe ich?",
    ),
]
FILE_PAIRS = [
    ("Can you check the size of {} in bytes?", "Now count the lines in {}."),
    ("¿Cuántos bytes ocupa {}?", "¿Cuántas líneas tiene {}?"),
    ("Quelle est la taille de {} en octets ?", "Combien de lignes contient {} ?"),
    ("Wie viele Bytes hat {}?", "Wie viele Zeilen enthält {}?"),
]


def service(group, a, b, opening, follow):
    packages = group == "packages"
    calls = (
        [{"name": "package_version", "arguments": {"distribution": v}} for v in (a, b)]
        if packages
        else [
            {"name": "file_bytes", "arguments": {"path": a}},
            {"name": "file_lines", "arguments": {"path": b}},
        ]
    )
    row = dict(
        id=sha(a + " " + b),
        family=group,
        questions=[opening.format(a), follow.format(b)],
        calls=calls,
    )
    if not packages:
        row["paths"] = [a, b]
    funcs = {t.name: t for t in catalog(row)}
    row["answers"] = [
        str(next(iter(funcs[c["name"]].callback(**c["arguments"]).values())))
        for c in calls
    ]
    assert len(set(row["answers"])) == 2
    return row


def main():
    tok = load_tokenizer(local_snapshot())
    packages = package_tools.snapshot()
    paths = sorted(
        str(p)
        for p in map(
            Path,
            subprocess.check_output(
                ["git", "ls-tree", "-r", "--name-only", PRIOR, "research/experiments"],
                text=True,
            ).splitlines(),
        )
        if p.stem.startswith(
            ("attention-", "register-", "induction-", "loop-", "cache-")
        )
    )
    held_pkgs = [
        "numpy",
        "mlx",
        "transformers",
        "pypdf",
        "jsonschema",
        "packaging",
        "safetensors",
        "jinja2",
        "tokenizers",
        "requests",
        "mlx-lm",
        "pytest",
    ]
    eval_pkgs = list(zip(held_pkgs[::2], held_pkgs[1::2], strict=True))
    eval_files = list(zip(paths[:12:2], paths[1:12:2], strict=True))
    cases = []
    train = []
    for family, resources, forms in [
        ("packages", eval_pkgs, PAIRS),
        ("files", eval_files, FILE_PAIRS),
    ]:
        cases.extend(
            service(family, a, b, *forms[i % 4]) for i, (a, b) in enumerate(resources)
        )
    resources = sorted(set(packages) - set(held_pkgs))
    for i, a in enumerate(resources[:64]):
        b = resources[(i + 7) % len(resources)]
        if a != b and packages[a]["version"] != packages[b]["version"]:
            train.extend(
                service_targets(
                    service(
                        "packages",
                        a,
                        b,
                        "Check the version of {} that is installed.",
                        "Please look up {} too.",
                    )
                )
            )
    rows = external()
    excluded = former({q for _, q, _ in rows}, PRIOR)
    used = {tuple(norm(q)[:8]) for q in excluded}
    counts = Counter()
    held = []
    for family, q, gold in sorted(
        rows, key=lambda r: sha("published-function-test-91 " + r[1])
    ):
        key = tuple(norm(q)[:8])
        if not gold or key in used or counts[family] >= 12:
            continue
        used.add(key)
        counts[family] += 1
        held.append(grams(q))
        cases.append(
            dict(id=sha(q), family=family, questions=[q + INSTRUCTION], gold=gold)
        )
    assert sorted(counts.values()) == [12, 12]
    public = [
        dict(r, questions=[r["user_request"]])
        for r in read(DATA / "unified-session-panel.json")["rows"]
        if r["family"] == "public"
    ]
    for r in public:
        r.pop("initial_ids")
    assert len(public) == 22
    cases += public
    cases.sort(key=lambda r: sha("qwen-function-order-93 " + r["id"]))
    for r in cases:
        prefix = qwen_tools.render(
            tok,
            [dict(role="user", content=r["questions"][0])],
            qwen_tools.declarations(catalog(r)),
        )
        r["initial_ids"] = tok.encode(prefix, add_special_tokens=False)
    sources = {}
    for dataset, limit, cut in [
        ("tool-dialogue-typed-training", 400, 1074),
        ("tool-program-reasoning-training", 480, 1280),
        ("tool-sql-learning-training", 160, 1536),
    ]:
        path = DATA / (dataset + ".json")
        sources[str(path)] = digest(path)
        pool = read(path)["examples"][:cut]
        if dataset == "tool-program-reasoning-training":

            def clean(r):
                g = grams(r["messages"][1]["content"].removesuffix(INSTRUCTION))
                return all(len(g & h) / max(len(g | h), 1) < 0.4 for h in held)

            pool = [r for r in pool if clean(r)]
        selected = sorted(
            pool, key=lambda r: sha("published-function-fit-92 " + r["id"])
        )[:limit]
        for r in selected:
            e = from_compact(r)
            e["id"] = dataset + ":" + e["id"]
            train.append(e)
    assert len({r["id"] for r in train}) == len(train)
    fi = {
        p: dict(
            sha256=digest(Path(p)),
            bytes=Path(p).stat().st_size,
            lines=len(Path(p).read_bytes().splitlines()),
        )
        for pair in eval_files
        for p in pair
    }
    for row in train:
        token_pair(tok, row)
    artifacts = {
        DATA / "native-function-training.json": dict(sources=sources, examples=train),
        DATA / "native-function-panel.json": dict(
            excluded_through=PRIOR, rows=cases, files=fi, environment=packages
        ),
    }
    for path, material in artifacts.items():
        if path.exists() and read(path) != material:
            raise ValueError(
                f"Current environment differs from registered material at {path}"
            )
    for path, material in artifacts.items():
        if not path.exists():
            save(path, material)
    print(len(train), len(cases), Counter(r["family"] for r in cases), flush=True)


if __name__ == "__main__":
    main()
