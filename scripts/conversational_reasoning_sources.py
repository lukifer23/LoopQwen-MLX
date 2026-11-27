"""Select complete first-and-follow-up CoQA episodes with published evidence and independent dev answers."""

import argparse
import re
from collections import Counter
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from scripts.content_read_material import windows
from scripts.table_policy_material import persist

CACHE = Path("data/research_external/coqa")
OUT = Path("research/sources/conversational-reasoning/episodes.json")
HASH = {
    "train": "b0fdb2bc1bd38dd3ca2ce5fa2ac3e02c6288ac914f241ac409a655ffb6619fa6",
    "dev": "dfa367a9733ce53222918d0231d9b3bedc2b8ee831a2845f62dfc70701f2540a",
}
DOMAINS = ("wikipedia", "gutenberg")


def normalize(text):
    return " ".join(re.findall(r"[\w]+", text.casefold()))


def gather(split, download):
    name = f"coqa-{split}-v1.0.json"
    path = CACHE / name
    if download and (not path.is_file() or digest(path) != HASH[split]):
        with urlopen(
            "https://downloads.cs.stanford.edu/nlp/data/coqa/" + name + "?source=coqa",
            timeout=75,
        ) as f:
            raw = f.read()
        assert sha256(raw).hexdigest() == HASH[split]
        CACHE.mkdir(exist_ok=True, parents=True)
        path.write_bytes(raw)
    assert digest(path) == HASH[split]
    valid = []
    for w in read(path)["data"]:
        if (
            w["source"] not in DOMAINS
            or min(len(w["questions"]), len(w["answers"])) < 2
        ):
            continue
        qa = []
        for i in (0, 1):
            q, a = w["questions"][i], w["answers"][i]
            assert q["turn_id"] == a["turn_id"] == i + 1
            lo, hi = a["span_start"], a["span_end"]
            if (
                lo < 0
                or lo >= hi
                or w["story"][lo:hi] != a["span_text"]
                or not q["input_text"].strip()
            ):
                break
            target = normalize(a["input_text"])
            if not target or target in ("unknown", "yes", "no"):
                break
            raters = [a["input_text"]]
            for group in w.get("additional_answers", {}).values():
                if len(group) > i and group[i]["turn_id"] == i + 1:
                    raters.append(group[i]["input_text"])
            if split == "dev" and sum(normalize(r) == target for r in raters) < 2:
                break
            qa.append(
                dict(
                    turn=i + 1,
                    question=q["input_text"],
                    answers=raters,
                    span=[lo, hi],
                    source_text=fingerprint(a["span_text"]),
                )
            )
        if len(qa) != 2 or normalize(qa[0]["answers"][0]) == normalize(
            qa[1]["answers"][0]
        ):
            continue
        valid.append((w, qa))
    return sorted(valid, key=lambda x: fingerprint([445697, x[0]["id"]]))


def build(download):
    blocked = set()
    names = set()
    out = {}
    for split, quota in (("dev", 12), ("train", 64)):
        chosen = []
        n = Counter()
        eligible = gather(split, download)
        for w, qa in eligible:
            domain = w["source"]
            if n[domain] >= quota or w["filename"] in names:
                continue
            keys = windows(w["story"])
            if keys & blocked:
                continue
            names.add(w["filename"])
            blocked.update(keys)
            n[domain] += 1
            chosen.append(
                dict(
                    source=domain,
                    id=w["id"],
                    file=w["filename"],
                    story=fingerprint(w["story"]),
                    turns=qa,
                )
            )
            if len(chosen) == len(DOMAINS) * quota:
                break
        assert n == {k: quota for k in DOMAINS}, n
        out[split] = chosen
    persist(
        OUT,
        dict(
            hashes=HASH,
            source="https://stanfordnlp.github.io/coqa/",
            episodes=out,
            method="Take the first two consecutive human questions and responses only from the literature and Wikipedia domains. Require two distinct open verbal answers and exact indexed evidence spans. For development require at least two recorded independently supplied human answers including the primary to match under punctuation/case normalization on each turn. Select deterministically without source filenames or normalized 16-word sequences recurring across enrolled episodes including training and development. This selects source-agreement cases rather than estimating the difficulty of all CoQA. No model saw these development openings in this preparation. Public pretraining and overlap with other public training sources are unknown. Store passage identities and evidence offsets; original passage text remains in the verified publisher download.",
        ),
    )
    print({k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true")
    build(p.parse_args().download)
