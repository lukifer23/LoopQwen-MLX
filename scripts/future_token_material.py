"""Enroll independent works and manuals for a causal language-model comparison."""

import hashlib
import io
import json
import re
from pathlib import Path
from urllib.request import urlopen
from xml.etree import ElementTree
from zipfile import ZipFile

from mlx_lm.utils import load_tokenizer
from selective_raw_material import grams, identity

from loopqwen.data import retention3
from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot

ROOT = Path("research/data")
CACHE = Path("data/research_external/future-token")
GIT = "777489f9e09c8d0dd6b12f9d90de6376330577a2"
BOOKS = {
    "lewis-carroll_alices-adventures-in-wonderland": (
        "9e5eb58995d9fd2f609363d14786d8b0166bbbb9",
        "fiction",
    ),
    "mary-shelley_frankenstein": (
        "abfa474c5e1df00ae8458ed8495b83e7f8b795cc",
        "fiction",
    ),
    "robert-louis-stevenson_treasure-island": (
        "8bd59dd6644e732c748d17e89b7370aacda67d76",
        "fiction",
    ),
    "arthur-conan-doyle_the-adventures-of-sherlock-holmes": (
        "6c94b03802c2fb2059b7f1521665a601d640f77e",
        "fiction",
    ),
    "frederick-douglass_narrative-of-the-life-of-frederick-douglass": (
        "0aa2e6434280bd6b7dc3a29f40863b77eb338110",
        "memoir",
    ),
    "benjamin-franklin_the-autobiography-of-benjamin-franklin": (
        "130142cf2ce6bd7ab883e4e830c29aea5dfe11a8",
        "memoir",
    ),
    "henry-adams_the-education-of-henry-adams": (
        "5380ba71427c3825a735f24db7b3365087f608ec",
        "memoir",
    ),
    "john-muir_my-first-summer-in-the-sierra": (
        "a1e6609fa8b1c956d1bb52ad68571407800cf59c",
        "memoir",
    ),
}
MANUALS = ("status", "diff", "log", "worktree", "branch", "switch", "restore", "tag")
ARTICLES = (
    "Montana",
    "Solar_energy",
    "Frédéric_Chopin",
    "Portugal",
    "New_York_City",
    "Classical_music",
    "Alps",
    "Daylight_saving_time",
)


def fetch(key, url, sources):
    path = CACHE / key
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url, timeout=90) as f:
            payload = f.read()
        temporary = path.with_suffix(".part")
        temporary.write_bytes(payload)
        temporary.replace(path)
    sources[key] = {"url": url, "sha256": digest(path)}
    return path.read_bytes()


def book_paragraphs(blob):
    result = []
    with ZipFile(io.BytesIO(blob)) as z:
        for member in sorted(z.namelist()):
            if "/src/epub/text/" not in member or not member.endswith(".xhtml"):
                continue
            root = ElementTree.fromstring(z.read(member))
            body = root.find(".//{*}body")
            if body is None or "bodymatter" not in body.attrib.get(
                "{http://www.idpf.org/2007/ops}type", ""
            ):
                continue
            for i, el in enumerate(body.iter("{http://www.w3.org/1999/xhtml}p")):
                result.append((Path(member).name + f"#{i}", "".join(el.itertext())))
    return result


def main():
    tok = load_tokenizer(local_snapshot())
    old = []
    for name in (
        "selective-raw-training",
        "selective-raw-panel",
        "raw-distillation-test",
        "cache-language-test",
        "cache-language-train",
    ):
        old.extend(read(ROOT / (name + ".json"))["rows"])
    old.extend(
        json.loads(line)
        for line in (retention3.ROOT / "validation.jsonl").read_text().splitlines()
    )
    excluded = set()
    for w in old:
        excluded.update(grams(w["ids"]))
    sources = {}
    rows = []

    def enroll(source, category, paragraphs, count):
        order = sorted(
            paragraphs,
            key=lambda p: hashlib.sha256(
                ("future-tokens-9116" + p[1]).encode()
            ).digest(),
        )
        n = 0
        for location, paragraph in order:
            text = " ".join(paragraph.split())
            ids = tok.encode(text, add_special_tokens=False)[:129]
            if len(ids) < 64:
                continue
            pieces = grams(ids)
            if excluded & pieces:
                continue
            excluded.update(pieces)
            rows.append(
                {
                    "id": category + "-" + identity(ids)[:20],
                    "source": source,
                    "category": category,
                    "location": location,
                    "ids": ids,
                }
            )
            n += 1
            if n == count:
                return
        raise ValueError((source, n, count))

    for source, (revision, category) in BOOKS.items():
        url = f"https://codeload.github.com/standardebooks/{source}/zip/{revision}"
        blob = fetch(source + ".zip", url, sources)
        enroll(source, category, book_paragraphs(blob), 8)
    for name in MANUALS:
        source = "git-" + name
        url = f"https://raw.githubusercontent.com/git/git/{GIT}/Documentation/{source}.txt"
        text = fetch(source + ".txt", url, sources).decode()
        paras = re.split(r"\n\s*\n", text)
        # Consecutive prose paragraphs keep short manual entries with their context.
        snippets = [
            (str(i), "\n\n".join(paras[i : i + 3])) for i in range(0, len(paras), 3)
        ]
        enroll(source, "manual", snippets, 4)
    squad = Path("data/research_external/squad-train-v1.1.json")
    if not squad.exists():
        from urllib.request import urlretrieve

        urlretrieve(
            "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json", squad
        )
    sources[squad.name] = {
        "url": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
        "sha256": digest(squad),
        "license": "CC BY-SA 4.0",
    }
    articles = {a["title"]: a for a in read(squad)["data"]}
    for title in ARTICLES:
        paragraphs = [
            (str(i), p["context"]) for i, p in enumerate(articles[title]["paragraphs"])
        ]
        enroll(title, "encyclopedia", paragraphs, 4)
    assert len(rows) == 128 and len({r["source"] for r in rows}) == 24
    save(ROOT / "future-token-panel.json", {"sources": sources, "rows": rows})
    print(
        "Enrolled",
        len(rows),
        "passages across",
        len({r["source"] for r in rows}),
        "sources",
    )


if __name__ == "__main__":
    main()
