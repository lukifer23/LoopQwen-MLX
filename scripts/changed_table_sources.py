"""Reconstruct unanimous answers to the same human question on two actual published tables."""

import argparse
import csv
import hashlib
import io
import re
import shutil
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save

CACHE = Path("data/research_external/changed-tables")
DEST = Path("research/sources/changed-tables")
REV = "7d455a5a707b96341ef72aff9428749d443d8aa9"
DPD = "https://downloads.cs.stanford.edu/nlp/software/sempre/wikitable/dpd/"
FILES = {
    "original.zip": (
        f"https://codeload.github.com/ppasupat/WikiTableQuestions/zip/{REV}",
        "7a9a29a784a8b616d27e08b552efa3673fca1b89dffc3d407332ee9049b6c5d0",
    ),
    "turk.tar.gz": (
        DPD + "turk.tar.gz",
        "9b7f96c9f5e3446286f37d91bd8013109ad84c135d81aad486096d7d86740357",
    ),
    "altered.tar.gz": (
        DPD + "altered.tar.gz",
        "5c249ad91ba2219cad89ee977afe218d435f8ccd5dc6013123566479873d52a6",
    ),
}
LIMITS = {"train": 256, "dev": 24}


def norm(value):
    return " ".join(value.casefold().split())


def labels():
    votes = defaultdict(dict)
    with tarfile.open(CACHE / "turk.tar.gz") as archive:
        for entry in archive:
            if not entry.isfile() or not Path(entry.name).name.isdigit():
                continue
            lines = io.TextIOWrapper(archive.extractfile(entry), encoding="utf-8")
            for row in csv.reader(lines, delimiter="\t", quoting=csv.QUOTE_NONE):
                _, key, world, label, answer, *ratings = row
                if (
                    not re.fullmatch(r"A[3-6]", label)
                    or not answer
                    or len(answer) > 72
                    or "|" in answer
                ):
                    continue
                responses = [
                    v
                    for who, v in zip(ratings[::2], ratings[1::2], strict=True)
                    if who != "GOLD"
                ]
                if len(responses) < 3 or any(
                    norm(v) != norm(answer) for v in responses
                ):
                    continue
                votes[key][int(world)] = {"answer": answer, "votes": responses}
    return votes


def table(raw, sep):
    # The original CSV is quoted. The altered release is plain TSV with escaped cells.
    args = {} if sep == "," else {"quoting": csv.QUOTE_NONE}
    rows = list(
        csv.reader(io.StringIO(raw.decode("utf-8"), newline=""), delimiter=sep, **args)
    )
    if not 3 <= len(rows) <= 37 or not 2 <= len(rows[0]) <= 12 or len(raw) > 7200:
        return None
    header = rows[0]
    if any(not x.strip() for x in header) or len({norm(x) for x in header}) != len(
        header
    ):
        return None
    if any(len(row) != len(header) for row in rows):
        return None
    if any("\\" in cell or "\x00" in cell for row in rows for cell in row):
        return None
    return rows


def build(fetch=False):
    CACHE.mkdir(exist_ok=True, parents=True)
    for filename, (url, sha) in FILES.items():
        path = CACHE / filename
        if fetch and not path.is_file():
            temp = path.with_suffix(".partial")
            with urlopen(url, timeout=120) as src, temp.open("wb") as out:
                shutil.copyfileobj(src, out)
            assert digest(temp) == sha
            temp.replace(path)
        assert digest(path) == sha
    votes = labels()
    candidates = {}
    order = {}
    with ZipFile(CACHE / "original.zip") as z:
        prefix = f"WikiTableQuestions-{REV}/"
        for split, limit in LIMITS.items():
            source = f"data/random-split-1-{split}.tsv"
            questions = csv.DictReader(
                io.StringIO(z.read(prefix + source).decode()),
                delimiter="\t",
                quoting=csv.QUOTE_NONE,
            )
            pool = []
            for q in questions:
                key = q["id"]
                gold = q["targetValue"]
                world = votes.get(key, {})
                if (
                    0 not in world
                    or norm(world[0]["answer"]) != norm(gold)
                    or not any(
                        i and norm(w["answer"]) != norm(gold) for i, w in world.items()
                    )
                ):
                    continue
                raw = z.read(prefix + q["context"])
                data = table(raw, ",")
                if data is None:
                    continue
                cases = [
                    i for i, w in world.items() if i and norm(w["answer"]) != norm(gold)
                ]
                cases.sort(key=lambda i: fingerprint([86537, key, i]))
                pool.append(
                    {
                        "id": key,
                        "question": q["utterance"],
                        "context": q["context"],
                        "split": split,
                        "worlds": cases,
                        "original": {
                            "world": 0,
                            "rows": data,
                            **world[0],
                            "sha256": hashlib.sha256(raw).hexdigest(),
                        },
                    }
                )
            pool.sort(key=lambda q: fingerprint([86537, q["context"], q["id"]]))
            # Only one question per source table is enrolled. Process a deterministic reserve
            # for malformed, oversize or nonmatching alternate table headers.
            unique = {}
            for q in pool:
                unique.setdefault(q["context"], q)
            order[split] = [q["id"] for q in unique.values()][: limit * 2]
            candidates.update(
                {q["id"]: q for q in unique.values() if q["id"] in order[split]}
            )
    changed = defaultdict(dict)
    with tarfile.open(CACHE / "altered.tar.gz", mode="r|gz") as z:
        for member in z:
            if len(z.members) > 1000:
                z.members.clear()
            if not member.isfile():
                continue
            _, key, tail = member.name.split("/")
            if key not in candidates or not tail.endswith(".tsv"):
                continue
            i = int(tail[:-4])
            q = candidates[key]
            if i not in q["worlds"]:
                continue
            raw = z.extractfile(member).read()
            rows = table(raw, "\t")
            if (
                rows is None
                or rows[0] != q["original"]["rows"][0]
                or rows == q["original"]["rows"]
            ):
                continue
            changed[key][i] = {
                "world": i,
                "rows": rows,
                **votes[key][i],
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
    enrolled = {}
    for split, ids in order.items():
        seen = set()
        cohort = []
        for key in ids:
            q = candidates[key]
            if not changed[key] or norm(q["question"]) in seen:
                continue
            other = next(i for i in q["worlds"] if i in changed[key])
            orig = q["original"]
            cohort.append(
                {k: q[k] for k in ("id", "question", "context", "split")}
                | {"versions": [orig, changed[key][other]]}
            )
            seen.add(norm(q["question"]))
            if len(cohort) == LIMITS[split]:
                break
        assert len(cohort) == LIMITS[split], (split, len(cohort))
        enrolled[split] = cohort
    for key in ("context", "question", "id"):
        assert not {norm(q[key]) for q in enrolled["train"]} & {
            norm(q[key]) for q in enrolled["dev"]
        }
    counts = Counter()
    for cohort in enrolled.values():
        for w in cohort:
            left, right = w["versions"]
            assert (
                norm(left["answer"]) != norm(right["answer"])
                and left["rows"][0] == right["rows"][0]
            )
            for v in w["versions"]:
                counts["ratings"] += len(v["votes"])
                counts["rows"] += len(v["rows"]) - 1
    manifest = {
        "source_revision": REV,
        "files": {k: {"url": u, "sha256": s} for k, (u, s) in FILES.items()},
        "source_split": "WikiTableQuestions original random-split-1, disjoint tables. Alternate worlds from the authors' annotation release on the original training pool.",
        "counts": dict(counts),
        "cohorts": enrolled,
    }
    path = DEST / "originals.json"
    DEST.mkdir(exist_ok=True, parents=True)
    if path.exists():
        assert read(path) == manifest
    else:
        save(path, manifest)
    print(
        {k: len(v) for k, v in enrolled.items()}, dict(counts), digest(path), flush=True
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acquire", action="store_true")
    build(p.parse_args().acquire)
