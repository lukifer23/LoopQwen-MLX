"""Prepare independent naturally revised crate versions with nearby in-document distractors."""

import bisect
from pathlib import Path
import tomllib

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.qwen_tools import render
from loopqwen.research.cross_view_study import tokenizer
from scripts.long_branch_sources import CACHE, SPECS, OUT as PRIOR, cargo
from scripts.program_language_material import user
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

OUT = Path("research/sources/changed-crates/qualified.json")
SEED = 991733


def question(name):
    return f"We need to reproduce this build. What version of the crate {name} is locked in this file? Give the exact version."


def load():
    old = read(PRIOR)["sources"]["deno"]
    raw = [(CACHE / name).read_bytes() for name in SPECS["deno"][3]]
    rows = [cargo(x) for x in raw]
    protected = {w["key"] for d in old["documents"] for w in d["questions"]}
    keys = sorted(
        (
            k
            for k in rows[0].keys() & rows[1].keys()
            if rows[0][k][0] != rows[1][k][0] and k not in protected
        ),
        key=lambda k: fingerprint([SEED, k]),
    )
    assert len(keys) == 104
    return old, raw, rows, keys


def build():
    committed(PRIOR)
    old, raw, rows, keys = load()
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    datasets = []
    for v, (blob, mapping) in enumerate(zip(raw, rows, strict=True)):
        names = SPECS["deno"][3]
        assert digest(CACHE / names[v]) == old["documents"][v]["sha256"]
        text = blob.decode()
        body = "Here is the complete Cargo.lock file.\n\n" + text + "\nEnd of file.\n\n"
        template = render(tok, [user(body + question(keys[0]))], [])
        enc = tok._tokenizer(
            template, add_special_tokens=False, return_offsets_mapping=True
        )
        prefix_len = old["documents"][v]["common_prompt_tokens"]
        prefix = enc["input_ids"][:prefix_len]
        assert (
            fingerprint(prefix) == old["documents"][v]["common_ids"]
            and prefix_len > 131072
        )
        starts = [x for x, y in enc["offset_mapping"]]
        ends = [y for x, y in enc["offset_mapping"]]
        source_start = template.index(text)
        sorted_names = sorted(mapping, key=lambda k: mapping[k][1])
        verify = {
            w["name"]: w["version"]
            for w in tomllib.loads(text)["package"]
            if w["name"] in mapping
        }

        def span(name):
            val, begin, end = mapping[name]
            assert blob[begin:end].decode() == val == verify[name]
            lo = source_start + len(blob[:begin].decode())
            hi = lo + len(val)
            first = bisect.bisect_right(ends, lo)
            stop = bisect.bisect_left(starts, hi)
            assert first < stop <= prefix_len and val in tok.decode(prefix[first:stop])
            return dict(name=name, value=val, bytes=[begin, end], tokens=[first, stop])

        values = []
        for k in keys:
            i = sorted_names.index(k)
            # The foil is the nearest *different* real semver from another crate.
            others = [
                n
                for n in sorted_names[max(0, i - 20) : i + 21]
                if mapping[n][0] not in (rows[0][k][0], rows[1][k][0])
            ]
            foil = min(others, key=lambda n: (abs(mapping[n][1] - mapping[k][1]), n))
            # Keep every whole-prefix hash without encoding the same 147K rows
            # hundreds of times. The chat suffix begins at an actual tokenizer boundary.
            suffix = template[ends[prefix_len - 1] :].replace(
                question(keys[0]), question(k)
            )
            tail = tok.encode(suffix, add_special_tokens=False)
            joined = prefix + tail
            if k in (keys[0], keys[len(keys) // 2], keys[-1]):
                exact = tok.encode(
                    render(tok, [user(body + question(k))], []),
                    add_special_tokens=False,
                )
                assert exact == joined
            values.append(
                dict(
                    **span(k),
                    question=question(k),
                    foil=span(foil),
                    input_tokens=len(joined),
                    input_sha256=fingerprint(joined),
                    tail=tail,
                )
            )
        datasets.append(
            dict(
                source=old["documents"][v]["sha256"],
                url=old["documents"][v]["url"],
                prefix=old["documents"][v]["common_ids"],
                prefix_tokens=prefix_len,
                rows=values,
            )
        )
    assert all(
        a["name"] == b["name"]
        and a["question"] == b["question"]
        and a["value"] != b["value"]
        for a, b in zip(datasets[0]["rows"], datasets[1]["rows"], strict=True)
    )
    summary = dict(
        prior=digest(PRIOR),
        seed=SEED,
        training=keys[:80],
        reserved=keys[80:],
        documents=datasets,
        protocol="Exclude all twelve preselected Deno packages from the earlier closed study. The remaining 104 versions each change naturally between two complete upstream lockfiles. Before training, hash-split 80 package names for supervision and 24 for independent within-project source transfer. Prompt and package are identical across releases. Each version has an exact source byte and token range, and a separate nearby real crate carrying a different distracting version. Six complete independent prompt encodings verify the optimization for constructing the other input hashes. All 208 source and distractor values resolve with TOML and byte-level extraction. No target is inserted into a source document. The reserved subset is a narrow source-transfer check and must not be tuned on.",
    )
    persist(OUT, summary)
    print(
        "changed crates",
        len(keys),
        "late evidence",
        sum(r["tokens"][0] > 131072 for d in datasets for r in d["rows"]),
        flush=True,
    )


if __name__ == "__main__":
    build()
