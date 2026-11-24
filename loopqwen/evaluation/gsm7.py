"""Reserve publisher train questions excluded by the historical annotated-trace reader."""

import json
from pathlib import Path
from urllib.request import urlopen
from hashlib import sha256
from ..data.build_data import GSM_REV
from ..identity import fingerprint
from ..io import save, digest


def reserved(root):
    path = Path("data/v2/gsm_source.jsonl")
    url = f"https://raw.githubusercontent.com/openai/grade-school-math/{GSM_REV}/grade_school_math/data/train.jsonl"
    with urlopen(url, timeout=30) as f:
        reference = sha256(f.read()).hexdigest()
    if digest(path) != reference:
        raise ValueError("Publisher file digest differs")
    exclude = set()
    for training in Path("data").glob("*/train.jsonl"):
        with training.open() as file:
            for line in file:
                r = json.loads(line)
                q = r.get("question", r.get("prompt", "")).split("\nAnswer:")[0].strip()
                if q:
                    exclude.add(q)
    xs = []
    for line in path.open():
        r = json.loads(line)
        q = r["question"].strip()
        if q in exclude or "<<" in r["answer"]:
            continue
        ans = r["answer"].rsplit("####", 1)[-1].strip().replace(",", "")
        xs.append(dict(id=fingerprint(q), family="gsm8k", question=q, gold=ans))
    selected = sorted(xs, key=lambda r: fingerprint([71109, r["id"]]))[:12]
    if len(selected) != 12:
        raise ValueError("Too few reserved questions")
    save(
        root / "gsm-provenance.json",
        dict(
            url=url,
            revision=GSM_REV,
            sha256=reference,
            license="MIT",
            selection="Unannotated train-only; excluded from optional annotation path. Exact question overlap removed from local training files.",
            eligible=len(xs),
            selected=[r["id"] for r in selected],
        ),
    )
    return selected
