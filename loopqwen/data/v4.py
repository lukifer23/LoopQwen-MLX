"""Tokenized development mixture with closure-unique DAGs and verified pointer paths."""

import json
import random
import string
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from transformers import AutoTokenizer

from .. import REVISION
from ..identity import fingerprint
from ..io import read, save, digest
from ..model_loader import local_snapshot
from . import transitions, v2
from .graphs import solve

ROOT = Path("data/v4")


def dag(rng, shift):
    roots = rng.randint(2, 4)
    parents = {i: [] for i in range(roots)}
    tips = list(parents)
    for j in range(rng.randint(2, 5)):
        v = len(parents)
        idx = rng.randrange(len(tips))
        parents[v] = [tips[idx]]
        tips[idx] = v
    # Cascade and fan-in motifs occur in training. Topology OOD includes
    # a fan-out from the first join followed by reconvergence.
    left = tips.pop()
    while tips:
        v = len(parents)
        parents[v] = [left, tips.pop()]
        if tips and rng.random() < 0.5:
            parents[v].append(tips.pop())
        left = v
    if shift in ("topology", "combined"):
        j = len(parents)
        parents[j] = [left, 0]
        parents[j + 1] = [left, 1]
        parents[j + 2] = [j, j + 1]
        left = j + 2
    closure = set(parents)
    parents[len(parents)] = [rng.randrange(roots)]
    # Distinct durations canonicalize node names; value ranges separate V3 DAGs.
    start = 301 if shift in ("operand", "combined") else 41
    durations = rng.sample(range(start, start + 44), len(parents))
    names = rng.sample(string.ascii_uppercase, len(parents))
    ds = dict(zip(names, durations))
    ps = {names[i]: [names[u] for u in pr] for i, pr in parents.items()}
    target = names[left]
    times, depth, _ = solve(ds, ps, target)
    canon = lambda vertices: [
        durations[left],
        sorted(
            (durations[i], sorted(durations[p] for p in parents[i])) for i in vertices
        ),
    ]
    display = list(ds)
    rng.shuffle(display)
    question = (
        "Parallel jobs (duration after prerequisites): "
        + "; ".join(f"{v}: {ds[v]} after {','.join(ps[v]) or 'none'}" for v in display)
        + f". When does {target} finish?"
    )
    trace = [
        f"job {v} finishes at {times[v] - ds[v]}+{ds[v]} = {times[v]}" for v in times
    ]
    return dict(
        id=fingerprint(["v4-dag", canon(parents)]),
        closure=fingerprint(canon(closure)),
        question=question,
        answer=str(times[target]),
        trace="\n".join(trace + [f"Final answer: {times[target]}"]),
        category="dependencies",
        steps=len(trace),
        complexity=depth,
        shift=shift,
        template="parallel-dag",
        graph=dict(durations=ds, predecessors=ps, target=target),
        verification_method="topological DP and exhaustive path enumeration",
    )


def pointer(row, n):
    coords = [row["start"]] + row["states"][:n]
    trace = [f"Next({a}) = {b}" for a, b in zip(coords, coords[1:])]
    return dict(
        id=row["id"],
        question=transitions.render(row, n).removesuffix("\nDestination:"),
        answer=str(coords[-1]),
        trace="\n".join(trace + [f"Final answer: {coords[-1]}"]),
        category="pointer",
        complexity=n,
        steps=n,
        template="map-follow",
        transition=dict(mapping=row["map"], start=coords[0], n=n),
        verification_method="lookup and independent one-hot propagation",
    )


def build():
    if ROOT.exists():
        manifest = read(ROOT / "manifest.json")
        for k, rec in manifest["splits"].items():
            if digest(ROOT / (k + ".jsonl")) != rec["sha256"]:
                raise ValueError("Changed V4 split")
        return manifest
    trans = transitions.build()
    tok = AutoTokenizer.from_pretrained(local_snapshot())
    manifest = dict(
        schema=4,
        revision=REVISION,
        seed=4204,
        generator=digest(Path(__file__)),
        sources=dict(
            procedural="Project MIT",
            parent=digest(v2.ROOT / "manifest.json"),
            pointers=trans,
        ),
        splits={},
    )
    seen = set()
    closures = set()
    with TemporaryDirectory(dir=ROOT.parent) as folder:
        tmp = Path(folder) / "dataset"
        tmp.mkdir()
        for split, num in (("train", 7500), ("validation", 300), ("ood", 300)):
            rng = random.Random("4204/" + split)
            source = [
                json.loads(line) for line in (v2.ROOT / (split + ".jsonl")).open()
            ]
            ptr = [
                json.loads(line)
                for line in (
                    transitions.ROOT
                    / (("train" if split == "train" else "development") + ".jsonl")
                ).open()
            ]
            if split == "ood":
                ptr = ptr[192:]
            rng.shuffle(source)
            rng.shuffle(ptr)
            cats = Counter()
            supers = Counter()
            with (tmp / (split + ".jsonl")).open("w") as f:
                for i in range(num):
                    if i % 5 < 3:
                        r = source.pop()
                    elif i % 5 == 3:
                        for _ in range(256):
                            r = dag(
                                rng,
                                ("operand", "topology", "combined")[(i // 5) % 3]
                                if split == "ood"
                                else "id",
                            )
                            if r["closure"] not in closures:
                                break
                        else:
                            raise RuntimeError("Duplicate graph closure")
                        closures.add(r["closure"])
                    else:
                        while ptr and ptr[-1]["id"] in seen:
                            ptr.pop()
                        r = pointer(ptr.pop(), 1 + (i // 5) % 3)
                    if r["id"] in seen:
                        raise ValueError("Repeated data instance")
                    seen.add(r["id"])
                    if "variants" not in r:
                        r["variants"] = {
                            mode: v2.encode_protocol(r, tok, mode == "trace")
                            for mode in ("trace", "answer")
                        }
                    cats[r["category"]] += 1
                    supers.update(
                        {k: v["output_length"] for k, v in r["variants"].items()}
                    )
                    f.write(json.dumps(r, separators=(",", ":")) + "\n")
            manifest["splits"][split] = dict(
                count=num,
                sha256=digest(tmp / (split + ".jsonl")),
                supervised=dict(supers),
                categories=dict(cats),
            )
        save(tmp / "manifest.json", manifest)
        tmp.rename(ROOT)
    return manifest


if __name__ == "__main__":
    print(build()["splits"])
