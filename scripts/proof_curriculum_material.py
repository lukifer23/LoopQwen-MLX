"""Replay signed open-world Horn rules and construct source-indexed proof supervision."""

import hashlib
import json
import re
from collections import Counter
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

from loopqwen.io import digest, read, save
from loopqwen.research.reasoned_evidence_data import choose

SOURCE = Path("data/research_external/proofwriter-V2020.12.3.zip")
MEMBER = "proofwriter-dataset-V2020.12.3/OWA/depth-3/meta-train.jsonl"
SHA = "bbc5694901e8306d0bd659aa1ad53ccfd02c201864f4b320ffa3777827d1fc26"
PANEL = Path("research/data/reasoned-evidence-panel.json")
DEST = Path("research/data/proof-curriculum-training.json")
VARS = {"someone", "something"}
ATOM = re.compile(r'\("([^"]+)" "([^"]+)" "([^"]+)" "([+-])"\)')


def inverse(atom):
    return (*atom[:3], "-" if atom[-1] == "+" else "+")


def replay(world):
    """Least fixed point of the publisher's signed, function-free single-variable rules."""
    facts = {}
    names = set()
    rules = []
    numbered = {}
    for i, (key, row) in enumerate(
        [*world["triples"].items(), *world["rules"].items()], 1
    ):
        numbered[key] = i
        form = row["representation"].replace('"~"', '"-"')
        atoms = ATOM.findall(form)
        assert (
            form.count('"') == 8 * len(atoms)
            and atoms
            and all(a[-1] in ("-", "+") for a in atoms)
        )
        for sub, rel, obj, _ in atoms:
            names.add(sub)
            if rel != "is":
                names.add(obj)
        if key.startswith("triple"):
            assert len(atoms) == 1 and ATOM.fullmatch(form)
            facts[atoms[0]] = (key, ())
        else:
            assert form.count("->") == 1 and len(atoms) >= 2
            assert len(set(t for a in atoms for t in a) & VARS) <= 1
            rules.append((key, atoms[:-1], atoms[-1]))
    ground = []
    for key, body, head in rules:
        for name in sorted(names - VARS):
            atoms = [
                tuple(name if x in VARS else x for x in atom) for atom in [*body, head]
            ]
            ground.append((key, tuple(atoms[:-1]), atoms[-1]))
    while True:
        before = len(facts)
        # Derive a layer at a time so a first witness has minimum Horn depth.
        updates = {}
        for key, body, head in ground:
            if (
                head not in facts
                and head not in updates
                and all(a in facts for a in body)
            ):
                updates[head] = key, body
        facts.update(updates)
        if len(facts) == before:
            break
    assert not any(inverse(a) in facts for a in facts)
    return facts, ground, numbered


def render(atom):
    sub, rel, obj, sign = atom
    if rel == "is":
        return f"{sub} is {'not ' if sign == '-' else ''}{obj}"
    return (
        f"{sub} {rel} {obj}" if sign == "+" else f"it is false that {sub} {rel} {obj}"
    )


def trace(q, facts, ground, numbered):
    atom = ATOM.fullmatch(q["representation"]).groups()
    opposite = inverse(atom)
    label = "true" if atom in facts else "false" if opposite in facts else "uncertain"
    assert label == {True: "true", False: "false", "Unknown": "uncertain"}[q["answer"]]
    if label == "uncertain":
        blockers = {}
        for key, body, head in ground:
            if head in (atom, opposite):
                missing = next(x for x in body if x not in facts)
                blockers.setdefault(key, missing)
        sentences = [
            f"Rule [{numbered[k]}] would need evidence that {render(m)}."
            for k, m in list(blockers.items())[:2]
        ]
        if not blockers:
            sentences.append(f"No rule concludes either polarity of ‘{render(atom)}’.")
        sentences.append(
            f"Neither ‘{render(atom)}’ nor its opposite follows from the given observations."
        )
        return label, " ".join(sentences), [], len(blockers)
    target = atom if label == "true" else opposite
    steps = []
    seen = set()

    def walk(a):
        if a in seen:
            return
        key, body = facts[a]
        for parent in body:
            walk(parent)
        steps.append((numbered[key], a, [numbered[facts[p][0]] for p in body]))
        seen.add(a)

    walk(target)
    reason = " ".join(
        (
            f"[{i}] gives {render(a)}."
            if not parents
            else f"Applying [{i}] to "
            + " and ".join(f"[{j}]" for j in parents)
            + f" yields {render(a)}."
        )
        for i, a, parents in steps
    )
    return label, reason, steps, 0


def proofs():
    assert digest(SOURCE) == SHA
    with ZipFile(SOURCE) as archive, archive.open(MEMBER) as stream:
        worlds = [json.loads(line) for line in stream]
    worlds = choose(worlds, 617031, lambda w: w["id"])
    rows = []
    seen = set()
    counts = Counter()
    for w in worlds:
        if len(rows) == 192:
            break
        statements = [v["text"] for v in [*w["triples"].values(), *w["rules"].values()]]
        evidence = "\n".join(f"{i}. {v}" for i, v in enumerate(statements, 1))
        if len(evidence.split()) > 170 or evidence in seen:
            continue
        facts, ground, numbered = replay(w)
        for key, q in choose(w["questions"].items(), 617032, lambda x: x[0]):
            if q["answer"] != "Unknown" and q["QDep"] not in (1, 2, 3):
                continue
            label, reason, steps, blockers = trace(q, facts, ground, numbered)
            sign = ATOM.fullmatch(q["representation"]).group(4)
            if len(reason.split()) > 100 or counts[label, sign] == 32:
                continue
            counts[label, sign] += 1
            seen.add(evidence)
            rows.append(
                dict(
                    id=w["id"] + "-" + key,
                    evidence=evidence,
                    claim=q["question"],
                    label=label,
                    reason=reason,
                    theory=w["id"],
                    question=key,
                    polarity=sign,
                    depth=q["QDep"],
                    closure_size=len(facts),
                    witness=steps,
                    blocked_rules=blockers,
                    publisher_proof=q["proofs"],
                )
            )
            break
    assert len(rows) == len(seen) == 192 and set(counts.values()) == {32}
    return rows


def distributions(panel):
    held = {canonicalize_name(w["package"]) for w in panel["wheels"]}
    selected = {}
    for group in ("field-figure-dependencies-04", "field-paper-dependencies-01"):
        for p in sorted((Path("exports") / group).glob("*.whl")):
            with ZipFile(p) as z:
                data = z.read(
                    next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
                )
            head = BytesParser().parsebytes(data)
            name, version, spec = (
                head[k] for k in ("Name", "Version", "Requires-Python")
            )
            if (
                canonicalize_name(name) not in held
                and spec
                and re.fullmatch(r">=3\.\d+", spec)
            ):
                selected[name] = dict(
                    package=name,
                    release=version,
                    requirement=spec,
                    wheel_sha256=digest(p),
                    metadata_sha256=hashlib.sha256(data).hexdigest(),
                )
    rows = []
    versions = ("3.7.17", "3.8.18", "3.9.18", "3.10.13", "3.11.8", "3.12.2", "3.13.1")
    for w in selected.values():
        minimum = Version(w["requirement"][2:])
        for ver in versions:
            actual = Version(ver)
            ok = actual in SpecifierSet(w["requirement"])
            assert ok == (actual >= minimum)
            detail = "meets" if ok else "falls below"
            rows.append(
                dict(
                    id="published-" + w["package"] + "-" + ver,
                    evidence=w["requirement"],
                    claim=f"Python {ver} is allowed by {w['package']} {w['release']}.",
                    label="true" if ok else "false",
                    reason=f"{w['package']} requires Python {minimum} or later. Python {ver} {detail} that minimum.",
                    interpreter=ver,
                    **w,
                )
            )
    assert len(selected) >= 8 and min(Counter(w["label"] for w in rows).values()) >= 16
    return rows


def build():
    pool = proofs()
    panel = read(PANEL)
    practical = distributions(panel)
    human = read("research/data/reasoned-evidence-training.json")["human"]
    assert len(human) == 384
    data = dict(
        source_archive=SHA, member=MEMBER, human=human, proof=pool, package=practical
    )
    # Normalise tuples to the JSON representation for exact independent reconstruction.
    data = json.loads(json.dumps(data))
    if DEST.exists():
        assert read(DEST) == data
    else:
        save(DEST, data)
    print(
        "Verified author rationales, independently replayed logic worlds, actual distribution checks:",
        len(human),
        len(pool),
        len(practical),
    )


if __name__ == "__main__":
    build()
