import hashlib
import json
import random
from pathlib import Path
from urllib.request import urlopen

from transformers import AutoTokenizer

from .. import MODEL, REVISION
from ..io import read, save
from ..model_loader import local_snapshot
from .procedural_math import CATEGORIES, problem
from .tokenize import encode

GSM_REV = "3101c7d5072418e28b9008a6636bde82a006892c"
ROOT = Path("data/pilot")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_data(public=True, seed=41):
    rng = random.Random(seed)
    tok = AutoTokenizer.from_pretrained(str(local_snapshot()))
    seen = set()
    ROOT.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split, n in (("train", 4400), ("validation", 220), ("test", 220)):
        rows = []
        for j in range(n):
            for attempt in range(10000):
                row = problem(rng, CATEGORIES[j % len(CATEGORIES)])
                if row["id"] not in seen:
                    break
            else:
                raise RuntimeError("Procedural instance space exhausted")
            seen.add(row["id"])
            rows.append(encode(row, tok))
        splits[split] = rows
    sources = {
        "procedural": "Project-generated exact arithmetic, using integer, Decimal and Fraction operations."
    }
    if public:
        url = f"https://raw.githubusercontent.com/openai/grade-school-math/{GSM_REV}/"
        license_text = urlopen(url + "LICENSE", timeout=30).read().decode()
        if not license_text.startswith("MIT License"):
            raise ValueError("Unexpected GSM8K license")
        (ROOT / "GSM8K-LICENSE").write_text(license_text)
        sources["gsm8k"] = dict(
            url=url, revision=GSM_REV, license="MIT", format="Gold numeric answer only"
        )
        for source, dest, n in (("train", "train", 500), ("test", "gsm_test", 32)):
            rows = []
            with urlopen(
                url + f"grade_school_math/data/{source}.jsonl", timeout=45
            ) as stream:
                for i, line in enumerate(stream):
                    item = json.loads(line)
                    key = hashlib.sha256(item["question"].strip().encode()).hexdigest()
                    if key in seen:
                        continue
                    answer = (
                        item["answer"].rsplit("#### ", 1)[1].strip().replace(",", "")
                    )
                    r = encode(
                        dict(
                            id=f"gsm-{source}-{i}",
                            category="gsm8k",
                            steps=item["answer"].count("<<"),
                            prompt=item["question"] + "\nAnswer:",
                            answer=answer,
                        ),
                        tok,
                    )
                    if len(r["ids"]) > 65 and source == "train":
                        continue
                    seen.add(key)
                    rows.append(r)
                    if len(rows) == n:
                        break
            splits.setdefault(dest, []).extend(rows)
    manifest = dict(
        seed=seed,
        model=MODEL,
        revision=REVISION,
        sources=sources,
        splits={},
        preprocessing="Separate prompt and space-prefixed answer, append EOS, one document per row.",
    )
    for split, rows in splits.items():
        rng.shuffle(rows)
        path = ROOT / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        manifest["splits"][split] = dict(
            count=len(rows),
            sha256=digest(path),
            tokens=sum(len(r["ids"]) for r in rows),
        )
    save(ROOT / "manifest.json", manifest)
    return manifest


def load_split(split):
    if not (ROOT / "manifest.json").exists():
        raise FileNotFoundError(
            "Build the dataset with python -m loopqwen data --preset pilot"
        )
    manifest = read(ROOT / "manifest.json")
    path = ROOT / f"{split}.jsonl"
    if digest(path) != manifest["splits"][split]["sha256"]:
        raise ValueError(f"Dataset hash mismatch for {split}")
    return [json.loads(line) for line in path.read_text().splitlines()]
