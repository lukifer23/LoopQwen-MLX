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
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def build_data(public=True, seed=41, gsm_traces=False, retention_file=None):
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
    if gsm_traces and not public:
        raise ValueError("GSM8K traces require public data")
    if public:
        url = f"https://raw.githubusercontent.com/openai/grade-school-math/{GSM_REV}/"
        license_text = urlopen(url + "LICENSE", timeout=30).read().decode()
        if not license_text.startswith("MIT License"):
            raise ValueError("Unexpected GSM8K license")
        (ROOT / "GSM8K-LICENSE").write_text(license_text)
        sources["gsm8k"] = dict(
            url=url, revision=GSM_REV, license="MIT", format="Gold numeric answer only"
        )
        if gsm_traces:
            sources["gsm8k"]["format"] = (
                "Publisher solutions plus final answer on training rows"
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
                    if gsm_traces and source == "train":
                        answer = item["answer"].strip()
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
                    if (
                        len(r["ids"]) > (129 if gsm_traces else 65)
                        and source == "train"
                    ):
                        continue
                    seen.add(key)
                    rows.append(r)
                    if len(rows) == n:
                        break
            splits.setdefault(dest, []).extend(rows)
    if retention_file:
        from ..evaluation.loss_eval import PROSE

        path = Path(retention_file)
        existing = {
            r["prompt"].strip() for group in splits.values() for r in group
        } | set(PROSE)
        catalog = set()
        for line in path.read_text().splitlines():
            item = json.loads(line)
            source, license_name, text = (
                item[k].strip() for k in ("source", "license", "text")
            )
            if not source or not license_name or not text:
                raise ValueError(
                    "Every retention document requires text, source and license"
                )
            if text in existing:
                continue
            ids = tok.encode(text, add_special_tokens=False) + [tok.eos_token_id]
            if len(ids) < 2:
                continue
            key = hashlib.sha256(text.encode()).hexdigest()
            splits["train"].append(
                dict(
                    id="text-" + key,
                    category="language",
                    steps=0,
                    prompt="",
                    answer=text,
                    ids=ids,
                    prefix=1,
                )
            )
            catalog.add((source, license_name))
            existing.add(text)
        sources["retention"] = dict(
            sha256=digest(path),
            attributions=[dict(source=s, license=l) for s, l in sorted(catalog)],
        )
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
