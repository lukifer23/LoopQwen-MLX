"""A fixed answer-format-compatible slice of the original 500-problem test split."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from ..evaluation.math_eval import normalized
from ..io import save
from .tokenize import encode

REVISION = "7ecc794703b2877f63226f2477a49b34f9b25163"
SHA256 = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"


def numeric_slice(tokenizer, n, root=Path("data/math500")):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "test.jsonl"
    url = f"https://media.githubusercontent.com/media/openai/prm800k/{REVISION}/prm800k/math_splits/test.jsonl"
    if not path.exists():
        for project, revision, name in (
            ("openai/prm800k", REVISION, "PRM"),
            ("hendrycks/math", "985bdc1696e88e8643f081a0ff4719da39f2ae2a", "MATH"),
        ):
            license_text = (
                urlopen(
                    f"https://raw.githubusercontent.com/{project}/{revision}/LICENSE",
                    timeout=30,
                )
                .read()
                .decode()
            )
            if not license_text.startswith("MIT License"):
                raise ValueError("Unexpected MATH license")
            (root / (name + "-LICENSE")).write_text(license_text)
        raw = urlopen(url, timeout=30).read()
        if hashlib.sha256(raw).hexdigest() != SHA256:
            raise ValueError("MATH split digest mismatch")
        path.write_bytes(raw)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Cached MATH split digest mismatch")
    rows = []
    for line in path.read_text().splitlines():
        item = json.loads(line)
        answer = normalized(item["answer"])
        if answer is None or "[asy]" in item["problem"]:
            continue
        rows.append(
            encode(
                dict(
                    id=item["unique_id"],
                    category="math500",
                    steps="unannotated",
                    prompt=item["problem"] + "\nAnswer:",
                    answer=answer,
                ),
                tokenizer,
            )
        )
        if len(rows) == n:
            break
    save(
        root / "manifest.json",
        dict(
            url=url,
            sha256=SHA256,
            license="MIT",
            selected=[r["id"] for r in rows],
            selection="First scalar-numeric golds without Asymptote figures, in source order",
        ),
    )
    return rows
