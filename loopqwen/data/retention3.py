"""Independent development-only retention passages. No source enters the training mix."""

import json
import random
import re
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

from transformers import AutoTokenizer

from ..io import digest, read, save
from ..model_loader import local_snapshot

CPYTHON = "https://raw.githubusercontent.com/python/cpython/4a5632fbf9bf59477c540e3f53fa7cdbeea3e3f5/"
SOURCES = {
    "general_prose": (
        "https://raw.githubusercontent.com/GITenberg/The-Wind-in-the-Willows_27805/bc45c53da38fb2270461a86efe4ba39982b06589/",
        "27805.txt",
    ),
    "scientific_exposition": (
        "https://raw.githubusercontent.com/GITenberg/The-Chemical-History-of-a-Candle_14474/8e778424cd5462a2d553cc0c5c141846994039f2/",
        "14474.txt",
    ),
    "technical_explanation": (CPYTHON, "Doc/tutorial/controlflow.rst"),
    "reference": (CPYTHON, "Doc/library/math.rst"),
    "code_description": (CPYTHON, "Doc/howto/functional.rst"),
}
ROOT = Path("data/v3_retention")


def fetch(url):
    with urlopen(url, timeout=45) as response:
        return response.read()


def build():
    if ROOT.exists():
        m = read(ROOT / "manifest.json")
        if digest(ROOT / "validation.jsonl") != m["sha256"]:
            raise ValueError("Retention corpus changed")
        return m
    tmp = Path(tempfile.mkdtemp(dir=ROOT.parent, prefix=".retention-"))
    tok = AutoTokenizer.from_pretrained(local_snapshot())
    m = dict(usage="development evaluation only", sources={}, count=0)
    import hashlib

    try:
        (tmp / "PSF-LICENSE").write_bytes(fetch(CPYTHON + "LICENSE"))
        with (tmp / "validation.jsonl").open("w") as out:
            for domain, (url, file) in SOURCES.items():
                raw = fetch(url + file)
                text = raw.decode("utf-8-sig").replace("\r", "")
                if url != CPYTHON:
                    rights = fetch(url + "metadata.yaml").decode()
                    if "rights: Public domain in the USA." not in rights:
                        raise ValueError("Unexpected book rights")
                    begin = re.search(r"^\*{3}\s*START OF[^\n]*", text, re.I | re.M)
                    end = re.search(r"^\*{3}\s*END OF[^\n]*", text, re.I | re.M)
                    if not begin or not end:
                        raise ValueError("Book boundaries absent in " + file)
                    text = text[begin.end() : end.start()]
                candidates = []
                for paragraph in re.split(r"\n\s*\n", text):
                    paragraph = " ".join(paragraph.split())
                    if not 160 <= len(paragraph) <= 650 or paragraph.startswith(
                        ("..", ">>>")
                    ):
                        continue
                    ids = tok.encode(paragraph, add_special_tokens=False)
                    if 35 <= len(ids) <= 150:
                        candidates.append((paragraph, ids))
                random.Random("phase3/" + domain).shuffle(candidates)
                chosen = candidates[:24]
                if len(chosen) < 12:
                    raise ValueError("Too few passages in " + file)
                for passage, ids in chosen:
                    out.write(
                        json.dumps(
                            dict(
                                id=hashlib.sha256(passage.encode()).hexdigest(),
                                category=domain,
                                source=file,
                                ids=ids,
                                prefix=1,
                                input_length=len(ids) - 1,
                            )
                        )
                        + "\n"
                    )
                m["sources"][domain] = dict(
                    url=url + file,
                    sha256=hashlib.sha256(raw).hexdigest(),
                    rights="PSF License Version 2"
                    if url == CPYTHON
                    else "Public domain in USA",
                    count=len(chosen),
                )
                m["count"] += len(chosen)
        m["sha256"] = digest(tmp / "validation.jsonl")
        save(tmp / "manifest.json", m)
        tmp.rename(ROOT)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return m


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
