"""Collect changed answers to practical file questions on complete original 128K+ sources."""

import argparse
import bisect
import re
import tomllib
from collections import Counter
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen
from xml.etree import ElementTree
from xml.parsers import expat

from loopqwen.identity import fingerprint
from loopqwen.qwen_tools import render
from loopqwen.research.cross_view_study import tokenizer
from scripts.table_policy_material import persist

CACHE = Path("data/research_external/long-branch")
OUT = Path("research/sources/long-branch/qualification.json")
CLDR = (
    "e1d37acce5dae468c414172be53b666d58d45be8",
    "96e54b273d2da4ec6fbad350ba19ff23fa24a588",
)
DENO = (
    "fb4db333c37ec1242a6f80738510d214018fea53",
    "fbb5be31ac23ef40f51025d6aeaaa91d9f082e4a",
)
SPECS = {
    "fr": (
        "unicode-org/cldr",
        "common/main/fr.xml",
        CLDR,
        ("fr-46.xml", "fr-49.xml"),
        (
            "8f8a56993bc40ec9a32e8bf9bf2a585b16c42719e0c73f56614535152ddc58ea",
            "7aa6b2f783d3223fb7e8fd1b4e0f3782d9731f2f91f829e93747e20ac7b3c268",
        ),
    ),
    "es": (
        "unicode-org/cldr",
        "common/main/es.xml",
        CLDR,
        ("es-46.xml", "es-49.xml"),
        (
            "05e5b29fc098457a59b49eb2edc8c010b5b7cfacc549bb50405ae18973b045e5",
            "85eb95dea0f020b13f7f615b09c86ebe568d71fd71f59c3720e23a65b9cddf50",
        ),
    ),
    "deno": (
        "denoland/deno",
        "Cargo.lock",
        DENO,
        ("deno-2.7.lock", "deno-2.8.lock"),
        (
            "5c34db249c90bfc351d35a541e817b1f5d28a0e7779777a1503bdbec52b5efdc",
            "a702a7652db509c72a73b1e3e70868354e2f2b1900b37e6848692150186723b2",
        ),
    ),
}


def xml(raw):
    parser = expat.ParserCreate()
    stack = []
    entries = {}
    counts = Counter()

    def begin(name, attrs):
        # Keep the standard field selectors, excluding editorial review status.
        part = name + "".join(
            f"[@{k}='{v}']"
            for k, v in sorted(attrs.items())
            if k not in ("draft", "references")
        )
        stack.append((part, parser.CurrentByteIndex))

    def end(_):
        path = "/".join(s[0] for s in stack)
        _, first = stack.pop()
        stop = parser.CurrentByteIndex
        start = raw.find(b">", first, stop) + 1
        if start <= first or b"<" in raw[start:stop] or b"&" in raw[start:stop]:
            return
        val = raw[start:stop].decode()
        if (
            not val.strip()
            or len(val) > 120
            or "↑" in val
            or path.startswith("ldml/identity/")
        ):
            return
        counts[path] += 1
        entries[path] = (val, start, stop)

    parser.StartElementHandler = begin
    parser.EndElementHandler = end
    parser.Parse(raw, True)
    return {k: v for k, v in entries.items() if counts[k] == 1}


def cargo(raw):
    parsed = tomllib.loads(raw.decode())["package"]
    counts = Counter(p["name"] for p in parsed)
    out = {}
    for block in re.finditer(
        rb"(?ms)^\[\[package\]\]\n.*?(?=^\[\[package\]\]|\Z)", raw
    ):
        obj = tomllib.loads(block.group().decode())["package"][0]
        if counts[obj["name"]] != 1:
            continue
        m = re.search(rb'(?m)^version = "([^"]+)"$', block.group())
        assert m.group(1).decode() == obj["version"]
        out[obj["name"]] = (
            obj["version"],
            block.start() + m.start(1),
            block.start() + m.end(1),
        )
    return out


def run(download):
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    results = {}
    CACHE.mkdir(exist_ok=True, parents=True)
    for cohort, (repo, file, revs, names, hashes) in SPECS.items():
        docs = []
        for rev, name, sha in zip(revs, names, hashes, strict=True):
            url = f"https://raw.githubusercontent.com/{repo}/{rev}/{file}"
            p = CACHE / name
            if not p.exists() or download:
                with urlopen(url, timeout=60) as remote:
                    payload = remote.read()
                assert sha256(payload).hexdigest() == sha
                p.write_bytes(payload)
            raw = p.read_bytes()
            assert sha256(raw).hexdigest() == sha
            docs.append((raw, url, (cargo if cohort == "deno" else xml)(raw)))
        old, new = [d[2] for d in docs]
        keys = [k for k in old.keys() & new.keys() if old[k][0] != new[k][0]]
        # Sample the different positions of changing facts throughout the complete file.
        if cohort != "deno":
            trees = [ElementTree.fromstring(doc[0]) for doc in docs]
            keys = [
                k
                for k in keys
                if all(
                    len(t.findall("./" + k.removeprefix("ldml/"))) == 1 for t in trees
                )
            ]
        keys.sort(key=lambda k: old[k][1])
        n = min(12, len(keys))
        selected = [keys[round(j * (len(keys) - 1) / (n - 1))] for j in range(n)]
        assert n >= 6 and len(set(selected)) == n
        receipts = []
        for (raw, url, records), sha in zip(docs, hashes, strict=True):
            body = raw.decode()
            prefix = (
                f"Here is the complete {file} file.\n\n" + body + "\nEnd of file.\n\n"
            )
            cases = []
            shared = None
            for k in selected:
                answer, a, b = records[k]
                if cohort == "deno":
                    q = f"We need to reproduce this build. What version of the crate {k} is locked in this file? Give the exact version."
                else:
                    q = f"We are reviewing the locale data. What is the exact text value at {k}? Preserve the characters and spacing from the file."
                    # Check the byte range with a second parser and a structural selector.
                    matches = ElementTree.fromstring(raw).findall(
                        "./" + k.removeprefix("ldml/")
                    )
                    assert len(matches) == 1 and matches[0].text == answer
                chat = render(tok, [{"role": "user", "content": prefix + q}], [])
                if shared is None:
                    offset = chat.index(body)
                    enc = tok._tokenizer(
                        chat, add_special_tokens=False, return_offsets_mapping=True
                    )
                    starts = [s for s, _ in enc["offset_mapping"]]
                    ends = [e for _, e in enc["offset_mapping"]]
                    # The final newline can merge with succeeding text. Cache only rows strictly inside the file.
                    length = bisect.bisect_left(ends, offset + len(body))
                    shared = enc["input_ids"][:length]
                ids = tok.encode(chat, add_special_tokens=False)
                assert 131072 < len(ids) < 262144 and ids[:length] == shared
                first = offset + len(raw[:a].decode())
                last = offset + len(raw[:b].decode())
                assert chat[first:last] == answer
                lo = bisect.bisect_right(ends, first)
                hi = bisect.bisect_left(starts, last)
                assert 0 <= lo < hi <= length
                cases.append(
                    {
                        "key": k,
                        "question": q,
                        "answer": answer,
                        "source_bytes": [a, b],
                        "source_tokens": [lo, hi],
                        "input_tokens": len(ids),
                        "input_sha256": fingerprint(ids),
                    }
                )
            receipts.append(
                {
                    "url": url,
                    "sha256": sha,
                    "bytes": len(raw),
                    "common_prompt_tokens": length,
                    "common_ids": fingerprint(shared),
                    "questions": cases,
                }
            )
        assert all(
            x["question"] == y["question"] and x["answer"] != y["answer"]
            for x, y in zip(
                receipts[0]["questions"], receipts[1]["questions"], strict=True
            )
        )
        results[cohort] = {
            "changed_fields": len(keys),
            "selected": n,
            "documents": receipts,
        }
    persist(
        OUT,
        {
            "scope": "Training-source preparation for the unresolved full-context capability. Natural upstream edits in complete original documents. File questions were generated from schema paths or package identities. No manually inserted evidence, model generation, optimizer update or accuracy estimate. Original English and Italian held files and held unit questions are excluded.",
            "sources": results,
        },
    )
    print({k: v["selected"] for k, v in results.items()}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--download", action="store_true")
    run(p.parse_args().download)
