"""Assemble disjoint human entailments, genuine reasoning supervision and wheel metadata."""

import csv
import json
import re
from collections import Counter, defaultdict
from email.parser import BytesParser
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from packaging.specifiers import SpecifierSet

from loopqwen import qwen_tools
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.reasoned_evidence_data import bind_file, choose, table, text

NAME = "reasoned-evidence-01"
SPEC = Path("research/experiments") / (NAME + ".json")
ANSWERS = ("true", "false", "uncertain")


def wheel_inventory():
    wheels = []
    for path in sorted(Path("exports/field-dependencies-01").glob("*.whl")):
        with ZipFile(path) as z:
            header = BytesParser().parsebytes(
                z.read(
                    next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
                )
            )
        wheels.append(
            {
                "package": header["Name"],
                "version": header["Version"],
                "requirement": header["Requires-Python"],
                "wheel": path.name,
                "sha256": digest(path),
            }
        )
    return wheels


def build():
    s = read(SPEC)
    seed = s["sample_seed"]
    tok = tokenizer(s)
    paths = {k: Path(v["path"]) for k, v in s["sources"].items()}
    assert all(digest(v) == s["sources"][k]["sha256"] for k, v in paths.items())

    def grouped(path):
        stories = defaultdict(list)
        for raw in path.open():
            r = json.loads(raw)
            premises = tuple(r["premises"])
            if (
                len(set(premises)) == len(premises)
                and all(p.strip() for p in premises)
                and r["conclusion"].strip()
            ):
                label = r["label"].lower()
                label = "uncertain" if label == "unknown" else label
                assert label in ANSWERS
                stories[premises].append((r["conclusion"], label))
        return stories

    original = grouped(paths["folio_train"])
    dev = grouped(paths["folio_validation"])

    def words(x):
        return " ".join(re.findall(r"\w+", x.casefold()))

    eligible = [
        p for p, q in dev.items() if {t[1] for t in q} == set(ANSWERS) and len(p) <= 15
    ]
    held = choose(eligible, seed, list)[: s["test_stories"]]
    forbidden = {words(x) for p in held for x in p}
    dead_claims = {words(q) for p in held for q, _ in dev[p]}

    def paragraph(p):
        return "\n".join(f"{i}. {v.strip()}" for i, v in enumerate(p, 1))

    teacher = []
    seen = set()
    counts = Counter()
    for p in choose(original, seed, list):
        if any(words(x) in forbidden for x in p) or p in held:
            continue
        for claim, label in choose(original[p], seed, list):
            if (
                counts[label] >= s["teacher_per_label"]
                or words(claim) in dead_claims
                or words(claim) in seen
            ):
                continue
            teacher.append(
                {
                    "id": fingerprint([p, claim])[:16],
                    "evidence": paragraph(p),
                    "claim": claim,
                    "label": label,
                }
            )
            seen.add(words(claim))
            counts[label] += 1
            break
    assert min(counts.values()) == s["teacher_per_label"] and set(counts) == set(
        ANSWERS
    )
    candidates = defaultdict(list)
    with paths["esnli"].open(newline="") as stream:
        for row in csv.DictReader(stream):
            label = {
                "entailment": "true",
                "contradiction": "false",
                "neutral": "uncertain",
            }.get(row["gold_label"])
            source, claim, reason = (
                row[k].strip() for k in ("Sentence1", "Sentence2", "Explanation_1")
            )
            if (
                label
                and 5 <= len(reason.split()) <= 45
                and source
                and claim
                and words(source) not in forbidden
                and words(claim) not in dead_claims
            ):
                candidates[label].append((row["pairID"], source, claim, reason))
    human = []
    identities = set()
    for label in ANSWERS:
        for ident, source, claim, reason in choose(
            candidates[label], seed, lambda r: r[0]
        ):
            if words(source) in identities:
                continue
            identities.add(words(source))
            human.append(
                {
                    "id": ident,
                    "evidence": source,
                    "claim": claim,
                    "label": label,
                    "reason": reason,
                }
            )
            if sum(v["label"] == label for v in human) == s["human_per_label"]:
                break
    assert len(human) == s["human_per_label"] * 3
    rows = []
    documents = {}

    def example(
        id, cluster, family, question, gold, document=None, call=None, obs=None, defs=()
    ):
        rows.append(
            {
                "id": id,
                "cluster": cluster,
                "family": family,
                "questions": [question],
                "gold": gold,
                "document": document,
                "calls": [call] if call else [],
                "observation": obs,
                "initial_ids": tok.encode(
                    qwen_tools.render(
                        tok, [{"role": "user", "content": question}], list(defs)
                    ),
                    add_special_tokens=False,
                ),
            }
        )

    with TemporaryDirectory() as work:
        tmp = Path(work)
        for i, p in enumerate(held):
            key = "Record " + fingerprint(p)[:8]
            context = paragraph(p)
            catalog = [
                (key, context),
                *(
                    (
                        ("Record " + fingerprint(held[(i + j) % len(held)])[:8]),
                        paragraph(held[(i + j) % len(held)]),
                    )
                    for j in (1, 2)
                ),
            ]
            raw = table(
                ["record", "observations"], choose(catalog, seed, lambda r: r[0])
            )
            name = f"records-{i}.csv"
            documents[name] = raw
            for label in ANSWERS:
                claim = next(q for q, y in choose(dev[p], seed, list) if y == label)
                fid = fingerprint([key, claim])[:16]
                example(fid + "-inline", key, "inline", text(context, claim), label)
                request = text(f'Read "{key}" from the attached file.', claim)
                defs, call, obs, _ = bind_file(tmp, name, raw, key, context, request)
                example(
                    fid + "-file", key, "file", request, label, name, call, obs, defs
                )
        wheels = wheel_inventory()
        assert len(wheels) == 44 and fingerprint(wheels) == s["wheel_inventory"]
        fname = "wheel-compatibility.csv"
        raw = table(
            ["distribution", "Requires-Python", "release"],
            [(w["package"], w["requirement"], w["version"]) for w in wheels],
        )
        documents[fname] = raw
        used = set()
        versions = ["3.8.20", "3.9.20", "3.10.15", "3.11.10"]
        for i in range(s["wheel_pairs"]):
            for version in (
                versions[i % len(versions) :] + versions[: i % len(versions)]
            ):
                options = choose(
                    [w for w in wheels if w["package"] not in used],
                    seed + i,
                    lambda w: w["package"],
                )
                pair = [
                    next(
                        (
                            w
                            for w in options
                            if (version in SpecifierSet(w["requirement"])) == f
                        ),
                        None,
                    )
                    for f in (True, False)
                ]
                if all(pair):
                    break
            assert all(pair)
            for w, label in zip(pair, ("true", "false"), strict=True):
                name = w["package"]
                used.add(name)
                q = text(
                    f'Read the published Requires-Python field for "{name}" in the attached wheel inventory.',
                    f"Python {version} satisfies the declared version constraint for {name} {w['version']}.",
                )
                defs, call, obs, _ = bind_file(
                    tmp, fname, raw, name, w["requirement"], q
                )
                example(
                    "wheel-" + name,
                    "wheel-pair-" + str(i),
                    "wheel",
                    q,
                    label,
                    fname,
                    call,
                    obs,
                    defs,
                )
    old = read("research/data/cross-view-panel.json")
    for w in old["rows"]:
        if w["family"] == "retention":
            documents[w["document"]] = old["documents"][w["document"]]
            rows.append(w)
    rows = choose(rows, seed, lambda w: w["id"])
    assert len(rows) == s["test_stories"] * 6 + s["wheel_pairs"] * 2 + 8
    for k, v in (
        ("material", {"rows": rows, "documents": documents, "wheels": wheels}),
        ("training", {"human": human, "teacher_pool": teacher}),
    ):
        path = Path(s[k])
        if path.exists():
            assert read(path) == v
        else:
            save(path, v)
    print(
        "Human explanations, teacher candidates, held requests:",
        len(human),
        len(teacher),
        len(rows),
    )


if __name__ == "__main__":
    build()
