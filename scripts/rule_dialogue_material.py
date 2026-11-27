"""Identify actual human dialogue trees in which changing one reply reverses the reference decision."""

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from loopqwen.identity import fingerprint
from loopqwen.io import digest, save

REVISION = "6e20dfe400a4da33aa9b671b29154dd4ce14dec4"
URL = f"https://raw.githubusercontent.com/sharc-data/sharc-data.github.io/{REVISION}/data/sharc1-official.zip"
SHA = "72dca3f4f3ba73b1d796b40e952a80d53cd2011ef90b2168b8bcaa818f5edd1e"
FOLDER = Path("research/sources/rule-dialogues")
CACHE = Path("data/research_external/sharc/sharc1-official.zip")
SEED = 451279
OUT_OF_SCOPE = re.compile(
    r"\b(?:gun|firearm|weapon|explosive|shooting|warfare|missile|military|combat|terroris\w*|pathogen|virus|biological agent|malware|cyberattack)\b",
    re.IGNORECASE,
)


def norm(s):
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def history(row):
    return tuple(
        (
            key.get("follow_up_question", key.get("followup_question")),
            key.get("follow_up_answer", key.get("followup_answer")),
        )
        for key in row["history"]
    )


def branches(rows):
    cases = defaultdict(list)
    for r in rows:
        h = history(r)
        if any(not q or not a for q, a in h):
            continue
        k = (
            r["tree_id"],
            r["source_url"],
            r["snippet"],
            r["question"],
            r["scenario"],
            h,
        )
        cases[k].append(r)
    paths = {}
    for key, parents in cases.items():
        root, prefix = key[:-1], key[-1]
        facts = " ".join([*root[1:], *(item for pair in prefix for item in pair)])
        if OUT_OF_SCOPE.search(facts):
            continue
        for parent in parents:
            question = parent["answer"]
            if OUT_OF_SCOPE.search(question) or not question.strip().endswith("?"):
                continue
            choices = []
            for observed in ("Yes", "No"):
                options = cases.get((*root, (*prefix, (question, observed))), [])
                verdict = {norm(r["answer"]) for r in options}
                if len(verdict) != 1 or not verdict.issubset({"yes", "no"}):
                    break
                r = min(options, key=lambda x: x["utterance_id"])
                choices.append(
                    {
                        "utterance_id": r["utterance_id"],
                        "last_user_reply": history(r)[-1][1],
                        "decision": r["answer"],
                    }
                )
            if len(choices) != 2 or norm(choices[0]["decision"]) == norm(
                choices[1]["decision"]
            ):
                continue
            kind = "agrees" if norm(choices[0]["decision"]) == "yes" else "reverses"
            ident = fingerprint([*root, prefix, question])
            path = {
                "id": ident,
                "tree": root[0],
                "url": root[1],
                "rule": root[2],
                "initial_question": root[3],
                "scenario": root[4],
                "history": [{"question": q, "answer": a} for q, a in prefix],
                "source_followup": question,
                "parent_utterance": parent["utterance_id"],
                "orientation": kind,
                "versions": choices,
            }
            if (
                ident not in paths
                or path["parent_utterance"] < paths[ident]["parent_utterance"]
            ):
                paths[ident] = path
    return list(paths.values())


def fields(w):
    return (
        ("url", norm(w["url"]).rstrip("/")),
        ("rule", norm(w["rule"])),
        ("question", norm(w["initial_question"])),
        ("tree", w["tree"]),
    )


def enroll(pool, n, excluded):
    chosen = []
    used = set(excluded)
    for orientation in ("reverses", "agrees"):
        picked = 0
        for row in sorted(pool, key=lambda r: fingerprint([SEED, r["id"]])):
            if row["orientation"] != orientation or used.intersection(fields(row)):
                continue
            chosen.append(row)
            used.update(fields(row))
            picked += 1
            if picked == n:
                break
        if picked != n:
            raise ValueError(
                f"Only {picked} distinct {orientation} rule/URL/question groups, require {n}"
            )
    return sorted(chosen, key=lambda r: fingerprint([SEED, "output", r["id"]]))


def build():
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(URL, timeout=60) as f:
            CACHE.write_bytes(f.read())
    assert digest(CACHE) == SHA
    with ZipFile(CACHE) as z:
        pools = {}
        all_rows = {}
        for split in ("train", "test"):
            rows = json.loads(z.read(f"sharc1-official/json/sharc_{split}.json"))
            all_rows[split] = {v["utterance_id"]: v for v in rows}
            pools[split] = branches(rows)
        held = enroll(pools["test"], 12, set())
        train = enroll(pools["train"], 48, {k for v in held for k in fields(v)})
        FOLDER.mkdir(parents=True, exist_ok=True)
        (FOLDER / "UPSTREAM-README.md").write_bytes(z.read("sharc1-official/README.md"))
    source = {"train": train, "test": held}
    for split, cohort in source.items():
        for w in cohort:
            par = all_rows[split][w["parent_utterance"]]
            assert par["answer"] == w["source_followup"]
            for v in w["versions"]:
                real = all_rows[split][v["utterance_id"]]
                assert history(real)[:-1] == history(par) and history(real)[-1] == (
                    w["source_followup"],
                    v["last_user_reply"],
                )
                assert all(
                    real[k] == par[k]
                    for k in (
                        "tree_id",
                        "source_url",
                        "snippet",
                        "question",
                        "scenario",
                    )
                )
                assert real["answer"] == v["decision"]
        assert len({norm(w["rule"]) for w in cohort}) == len(cohort)
    frozen = FOLDER / "dialogues.json"
    save(frozen, source)
    measured = {
        "url": URL,
        "revision": REVISION,
        "sha256": SHA,
        "license": "CC BY-SA 3.0",
        "paper": "https://aclanthology.org/D18-1233/",
        "seed": SEED,
        "source_rows": {k: len(v) for k, v in all_rows.items()},
        "eligible_branches": {
            k: dict(Counter(v["orientation"] for v in rows))
            for k, rows in pools.items()
        },
        "enrolled": {k: len(v) for k, v in source.items()},
        "orientations": {
            k: dict(Counter(r["orientation"] for r in v)) for k, v in source.items()
        },
        "shared_identifiers": {
            kind: len(
                {n for w in train for k, n in fields(w) if k == kind}
                & {n for w in held for k, n in fields(w) if k == kind}
            )
            for kind, _ in fields(held[0])
        },
        "source": digest(frozen),
    }
    assert not any(measured["shared_identifiers"].values())
    save(FOLDER / "qualification.json", measured)
    print(measured)


if __name__ == "__main__":
    build()
