"""Collect disjoint raw-text windows from encyclopedia, science and language documentation."""

import hashlib
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from mlx_lm.utils import load_tokenizer

from cache_language_material import excerpts
from loopqwen.data import retention3
from loopqwen.io import read, save, digest
from loopqwen.model_loader import local_snapshot
from loopqwen.research.projected_distillation import anchors

DATA = Path("research/data")
SCI = "2c94ad3e1aafab77146f384e23536f97a4849815"


def main():
    t = load_tokenizer(local_snapshot())
    seen = {
        r["id"].casefold()
        for k in ("train", "test")
        for r in read(DATA / ("cache-language-" + k + ".json"))["rows"]
    }
    h1 = excerpts("test", 20, seen, t)
    h2 = excerpts("validation", 32, seen, t)
    held = dict(
        sources=[{k: v for k, v in h.items() if k != "rows"} for h in (h1, h2)],
        rows=[
            dict(r, split=split)
            for h, split in [(h1, "test"), (h2, "validation")]
            for r in h["rows"]
        ],
    )
    wiki = excerpts("train", 384, seen, t)
    out = [dict(r, ids=r["ids"][:129]) for r in wiki["rows"]]
    cal = read("research/experiments/projected-distillation-01.json")
    raw = anchors(t, dict(cal, updates=128))
    out.extend(
        dict(
            id="cpython-" + str(i),
            category="documentation_and_literature",
            ids=r["ids"][:129],
        )
        for i, r in enumerate(raw)
    )
    pq_path = hf_hub_download(
        "allenai/sciq",
        "data/train-00000-of-00001.parquet",
        revision=SCI,
        repo_type="dataset",
    )
    science = pq.read_table(pq_path, columns=["support"]).column(0).to_pylist()
    squad_file = Path("data/research_external/squad-train-v1.1.json")
    if not squad_file.exists():
        from urllib.request import urlopen

        squad_file.parent.mkdir(exist_ok=True, parents=True)
        squad_file.write_bytes(
            urlopen(
                "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
                timeout=60,
            ).read()
        )
    squad = [p["context"] for a in read(squad_file)["data"] for p in a["paragraphs"]]

    def key(x):
        return hashlib.sha256(x.strip().encode()).hexdigest()

    for label, paragraphs in [("science", science), ("encyclopedia", squad)]:
        group = []
        for text in sorted(set(v for v in paragraphs if v and len(v) > 440), key=key):
            ids = t.encode(text.strip(), add_special_tokens=False)
            if len(ids) < 129:
                continue
            a = int(key(text)[:8], 16) % (len(ids) - 128)
            group.append(dict(id=key(text), category=label, ids=ids[a : a + 129]))
            if len(group) == 256:
                break
        if len(group) != 256:
            raise ValueError("Insufficient raw paragraphs")
        out.extend(group)
    # No exact 24-token spans shared with either held-out WikiText or the retained raw suite.
    import json

    retention3.build()
    validation = held["rows"] + [
        json.loads(s)
        for s in (retention3.ROOT / "validation.jsonl").read_text().splitlines()
    ]
    grams = {
        tuple(r["ids"][i : i + 24])
        for r in validation
        for i in range(len(r["ids"]) - 23)
    }
    before = len(out)
    out = [
        r
        for r in out
        if all(
            tuple(r["ids"][i : i + 24]) not in grams for i in range(len(r["ids"]) - 23)
        )
    ]
    random.Random(338287).shuffle(out)
    for r in out:
        rng = random.Random("targets/" + r["id"])
        r["positions"] = sorted(rng.sample(range(8, len(r["ids"]) - 1), 32))
    save(
        DATA / "raw-distillation-training.json",
        dict(
            rows=out,
            excluded=before - len(out),
            wiki_sha=wiki["source_sha256"],
            sciq_revision=SCI,
            sciq_sha=digest(Path(pq_path)),
            squad_sha=digest(squad_file),
            technical_sources=cal["anchor_files"],
        ),
    )
    save(DATA / "raw-distillation-test.json", held)
    print(len(out), len(validation), flush=True)


if __name__ == "__main__":
    main()
