"""Select human multi-fact table answers with two or more source-attested visible values."""

import csv
import io
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.table_worlds import SOURCE
from scripts.rule_dialogue_material import OUT_OF_SCOPE

REV = "bbc441be807212d736c8c35a18146b530dde11d0"
REMOTE = f"https://raw.githubusercontent.com/Yale-LILY/FeTaQA/{REV}/data/"
HASHES = {
    "train": "792d51fb53f813f31543b7a07cb29da7c28420a8ad4fdce9e156c2d7126bc661",
    "test": "3d4b9b1fd91bd5aef656b7bb8aa2a27ddb3a87c0c05afcc13018ae375ee0a6e4",
}
FOLDER = Path("research/sources/grounded-table-sentences")
CACHE = Path("data/research_external/fetaqa")
SEED = 463029


def norm(s):
    s = (
        unicodedata.normalize("NFKC", s)
        .casefold()
        .replace("−", "-")
        .replace("%", " percent ")
    )
    s = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", s)
    return " ".join(re.findall(r"(?<!\w)[+-]?\d+(?:\.\d+)?(?:[a-z]+)?|\w+|[£$€]", s))


def in_text(a, b):
    return bool(a) and (" " + a + " ") in (" " + b + " ")


def identity(r):
    return {
        ("page", norm(r["page"])),
        ("question", norm(r["question"])),
        ("table", fingerprint([[norm(c) for c in row] for row in r["rows"]])),
    }


def earlier():
    old = read(SOURCE)["cohorts"]
    rows = old["train"] + old["dev"]
    name = Path("data/research_external/changed-tables/original.zip")
    with ZipFile(name) as z:
        (member,) = [n for n in z.namelist() if n.endswith("/misc/table-metadata.tsv")]
        meta = {
            r["contextId"]: r["title"]
            for r in csv.DictReader(
                io.StringIO(z.read(member).decode()), delimiter="\t"
            )
        }
    blocked = set()
    for w in rows:
        for v in w["versions"]:
            blocked.update(
                identity(
                    {
                        "page": meta[w["context"]],
                        "question": w["question"],
                        "rows": v["rows"],
                    }
                )
            )
    return blocked


def eligible(r):
    cells = r["table_array"]
    w = len(cells[0]) if cells else 0
    if not (
        2 <= w <= 12 and 3 <= len(cells) <= 40 and all(len(row) == w for row in cells)
    ):
        return None
    if (
        any(not h.strip() for h in cells[0])
        or len({h.casefold().strip() for h in cells[0]}) != w
    ):
        return None
    if sum(len(s) for row in cells for s in row) > 6500 or OUT_OF_SCOPE.search(
        " ".join([r["table_page_title"], r["question"], r["answer"]])
    ):
        return None
    q, answer = norm(r["question"]), norm(r["answer"])
    if len(answer.split()) < 6 or len(answer.split()) > 75:
        return None
    candidates = []
    for i, j in r["highlighted_cell_ids"]:
        if not (0 < i < len(cells) and 0 <= j < w):
            continue
        text = cells[i][j]
        val = norm(text)
        if (
            len(val) < 2
            or val in {"no", "yes", "the", "none", "unknown", "not applicable"}
            or in_text(val, q)
            or not in_text(val, answer)
        ):
            continue
        candidates.append({"row": i, "column": j, "value": text, "normalized": val})
    atoms = []
    for c in sorted(
        candidates, key=lambda c: (-len(c["normalized"]), c["row"], c["column"])
    ):
        if not any(in_text(c["normalized"], a["normalized"]) for a in atoms):
            atoms.append(c)
    if len(atoms) < 2:
        return None
    return {
        "id": r["feta_id"],
        "page": r["table_page_title"],
        "url": r["page_wikipedia_url"],
        "section": r["table_section_title"],
        "source_table": r["table_source_json"],
        "rows": cells,
        "question": r["question"],
        "reference": r["answer"],
        "support_coordinates": r["highlighted_cell_ids"],
        "attested_values": atoms,
        "source_sha256": fingerprint(r),
    }


def build():
    pools = {}
    counts = {}
    blocked = earlier()
    for split, sha in HASHES.items():
        path = CACHE / (split + ".jsonl")
        if not path.is_file():
            CACHE.mkdir(parents=True, exist_ok=True)
            with urlopen(REMOTE + f"fetaQA-v1_{split}.jsonl", timeout=90) as f:
                path.write_bytes(f.read())
        assert digest(path) == sha
        source = [json.loads(line) for line in path.read_text().splitlines()]
        accepted = [v for r in source if (v := eligible(r)) is not None]
        allowed = [v for v in accepted if not identity(v).intersection(blocked)]
        counts[split] = {
            "published": len(source),
            "structurally_eligible": len(accepted),
            "after_prior_table_exclusion": len(allowed),
        }
        pools[split] = sorted(
            allowed, key=lambda r: fingerprint([SEED, split, r["id"]])
        )
    used = set(blocked)
    cohorts = {}
    for split, n in (("test", 24), ("train", 384)):
        out = []
        for r in pools[split]:
            ids = identity(r)
            if ids & used:
                continue
            used.update(ids)
            out.append(r)
            if len(out) == n:
                break
        assert len(out) == n
        cohorts[split] = out
    FOLDER.mkdir(parents=True, exist_ok=True)
    path = FOLDER / "originals.json"
    save(path, cohorts)
    facts = {}
    for split, rows in cohorts.items():
        n = Counter(len(r["attested_values"]) for r in rows)
        facts[split] = {
            "questions": len(rows),
            "distinct_pages": len({norm(r["page"]) for r in rows}),
            "complete_table_rows": sum(len(r["rows"]) - 1 for r in rows),
            "attested_values": sum(count * n_ for count, n_ in n.items()),
            "value_count_distribution": dict(sorted(n.items())),
        }
    info = {
        "paper": "https://aclanthology.org/2022.tacl-1.3/",
        "source": "https://github.com/Yale-LILY/FeTaQA",
        "revision": REV,
        "files": HASHES,
        "license": "CC BY-SA 4.0",
        "sample_seed": SEED,
        "source_populations": counts,
        "cohorts": facts,
        "selected": digest(path),
        "prior_changed_table_source": digest(SOURCE),
        "scope": "The selected questions have at least two different annotated cell values occurring as full token sequences in the human reference and absent from the question. Exact occurrence is a support diagnostic; a correct free-form explanation can paraphrase values, and unsupported extra claims can make an answer incorrect. All original questions, rows, responses and support coordinates are preserved.",
    }
    save(FOLDER / "qualification.json", info)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    build()
