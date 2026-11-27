"""Reserve changed dependency relationships in four complete, unmodified 128K+ release lockfiles."""

import argparse
import bisect
import itertools
import re
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.qwen_tools import render
from loopqwen.research.cross_view_study import tokenizer
from scripts.long_branch_sources import CACHE, OUT as TRAIN
from scripts.table_policy_material import persist

OUT = Path("research/sources/long-branch/dependency-relations.json")
FILES = {
    "foundry-rs/foundry": [
        (
            "v1.4.3",
            "fa9f934bdac4bcf57e694e852a61997dda90668a",
            "cbcded237f0a4c8d303ced74a2b63dfd36783548cd0cb68d1bdfe07976ab0224",
        ),
        (
            "v1.5.1",
            "b0a9dd9ceda36f63e2326ce530c10e6916f4b8a2",
            "3a29cfd00839c8c316ce79a0b6b7ae33d0a8a471e1a1e8289a61dc58eedcbe11",
        ),
    ],
    "anza-xyz/agave": [
        (
            "v2.3.0",
            "a2e21dda295b18dd09ac06542834c5e9b1b2dc84",
            "d9755fef2673aa458665930be26cdd8796acd7419039ebf999e4570de0444f8d",
        ),
        (
            "v3.0.0",
            "b6c96e84b10396b92912d4574dae7d03f606da26",
            "e3ce81d65b07dd11ca4e2e5f44227096ee435fbbdb31b735a378ce4de89cde9e",
        ),
    ],
}
SEED = 8249067
PREFIX = "Here is the complete Cargo.lock file.\n\n"
END = "\nEnd of file.\n\n"


def links(raw):
    pkgs = tomllib.loads(raw.decode())["package"]
    names = Counter(x["name"] for x in pkgs)
    versions = Counter((x["name"], x["version"]) for x in pkgs)
    edges = defaultdict(dict)
    blocks = list(
        re.finditer(rb"(?ms)^\[\[package\]\]\n.*?(?=^\[\[package\]\]|\Z)", raw)
    )
    assert len(blocks) == len(pkgs)
    for p, b in zip(pkgs, blocks, strict=True):
        assert tomllib.loads(b[0].decode())["package"] == [p]
        if names[p["name"]] != 1:
            continue
        resolved = defaultdict(list)
        for dep in p.get("dependencies", []):
            words = dep.split()
            if len(words) < 2 or names[words[0]] < 2 or versions[tuple(words[:2])] != 1:
                continue
            literal = ('"' + dep + '"').encode()
            if b[0].count(literal) != 1:
                continue
            start = b.start() + b[0].index(literal) + len(words[0].encode()) + 2
            end = start + len(words[1].encode())
            assert raw[start:end].decode() == words[1]
            resolved[words[0]].append((words[1], start, end))
        for dep, items in resolved.items():
            if len(items) == 1:
                edges[dep][p["name"]] = items[0]
    return edges


def request(a, b, dep):
    return f"We need to check whether this build contains a dependency version mismatch. Do {a} and {b} resolve {dep} to the same version? Give the two exact versions in that order and explain the comparison."


def build(download=False):
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    used = {
        w["key"] for w in read(TRAIN)["sources"]["deno"]["documents"][1]["questions"]
    }
    result = {}
    for repo, specs in FILES.items():
        docs = []
        all_edges = []
        for tag, rev, sha in specs:
            path = CACHE / (repo.split("/")[1] + "-" + tag + ".lock")
            url = f"https://raw.githubusercontent.com/{repo}/{rev}/Cargo.lock"
            if download or not path.exists():
                data = urlopen(url, timeout=40).read()
                from hashlib import sha256

                assert sha256(data).hexdigest() == sha
                path.write_bytes(data)
            assert digest(path) == sha
            raw = path.read_bytes()
            all_edges.append(links(raw))
            docs.append((url, path, raw))
        aa, bb = all_edges
        pool = []
        for dep in sorted(set(aa) & set(bb) - used):
            for a, b in itertools.combinations(
                sorted(aa[dep].keys() & bb[dep].keys() - used), 2
            ):
                old = aa[dep][a][0] == aa[dep][b][0]
                new = bb[dep][a][0] == bb[dep][b][0]
                if old != new:
                    pool.append((a, b, dep, old))
        selected = []
        deps = Counter()
        owners = set()
        direction = Counter()
        while len(selected) < 4:
            choices = [c for c in pool if not owners.intersection(c[:2])]
            if not choices:
                break
            c = min(
                choices,
                key=lambda z: (
                    direction[z[3]],
                    deps[z[2]],
                    fingerprint([SEED, repo, z]),
                ),
            )
            selected.append(c)
            owners.update(c[:2])
            deps[c[2]] += 1
            direction[c[3]] += 1
        assert len(selected) == 4 and direction == {True: 2, False: 2}, (repo, selected)
        entries = []
        for i, (url, path, raw) in enumerate(docs):
            text = raw.decode()
            chat = render(
                tok,
                [
                    {
                        "role": "user",
                        "content": PREFIX + text + END + request(*selected[0][:3]),
                    }
                ],
                [],
            )
            off = chat.index(text)
            encoded = tok._tokenizer(
                chat, add_special_tokens=False, return_offsets_mapping=True
            )
            starts = [a for a, b in encoded["offset_mapping"]]
            ends = [b for a, b in encoded["offset_mapping"]]
            n = bisect.bisect_left(ends, off + len(text))
            common = encoded["input_ids"][:n]
            assert 131072 <= n < 262144
            rows = []
            for a, b, dep, old in selected:
                vals = []
                evidence = []
                for name in (a, b):
                    v, s, e = all_edges[i][dep][name]
                    vals.append(v)
                    pos = off + len(raw[:s].decode())
                    start = bisect.bisect_right(ends, pos)
                    finish = bisect.bisect_left(starts, pos + len(v))
                    assert v in tok.decode(encoded["input_ids"][start:finish])
                    evidence.append(
                        dict(owner=name, bytes=[s, e], tokens=[start, finish])
                    )
                q = request(a, b, dep)
                ids = tok.encode(
                    render(
                        tok, [{"role": "user", "content": PREFIX + text + END + q}], []
                    ),
                    add_special_tokens=False,
                )
                assert ids[:n] == common and (vals[0] == vals[1]) == (
                    old if i == 0 else not old
                )
                rows.append(
                    dict(
                        owners=[a, b],
                        dependency=dep,
                        question=q,
                        versions=vals,
                        equal=vals[0] == vals[1],
                        evidence=evidence,
                        opening=len(ids),
                        input_sha256=fingerprint(ids),
                    )
                )
            entries.append(
                dict(
                    url=url,
                    sha256=digest(path),
                    file=path.name,
                    common_tokens=n,
                    common_ids=fingerprint(common),
                    cases=rows,
                )
            )
        result[repo] = dict(available_changed_relations=len(pool), documents=entries)
    obj = dict(
        seed=SEED,
        projects=result,
        method="Choose only uniquely named consumers with one explicit, uniquely resolvable edge to the multiply versioned dependency in both real releases. Require a changed equality answer, choose distinct consumers and balance the direction. Exclude enrolled Deno training identifiers. The model has no access to the parser or these records. No model outputs enter selection.",
    )
    persist(OUT, obj)
    print({k: [w["common_tokens"] for w in v["documents"]] for k, v in result.items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true")
    build(p.parse_args().download)
