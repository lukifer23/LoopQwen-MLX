"""Immutable map splits, counterfactual requests and independent pointer golds."""

from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import random

from . import transitions
from ..io import digest, read, save
from ..identity import fingerprint

ROOT = Path("data/phase5_pointer")
REG = Path("experiments/phase5.json")


def rows(split):
    with (ROOT / f"{split}.jsonl").open() as stream:
        return [json.loads(line) for line in stream]


def instance(mapping, start, order=None):
    map_id = hashlib.sha256(bytes(mapping)).hexdigest()
    return dict(
        map=mapping,
        map_id=map_id,
        start=start,
        example_id=fingerprint([map_id, start]),
        order=list(range(8)) if order is None else order,
        states=transitions.states(mapping, start, 6),
    )


def prompt(row, depth=None):
    links = " ".join(f"{i}>{row['map'][i]}" for i in row["order"])
    hop = "" if depth is None else f"  Hops: {depth}"
    return f"Map: {links}\nStart: {row['start']}{hop}\nState:"


def cycle(row):
    node = row["start"]
    for n in range(1, 9):
        node = row["map"][node]
        if node == row["start"]:
            return n
    raise ValueError("Not a permutation")


def prior(train, dev):
    report = []
    for j in range(3):
        hist = Counter(r["states"][j] for r in train)
        cond = [
            Counter(r["states"][j] for r in train if r["start"] == s) for s in range(8)
        ]
        majority = max(range(8), key=lambda s: hist[s])
        guess = [max(range(8), key=lambda s: h[s]) for h in cond]
        report.append(
            dict(
                chance=1 / 8,
                conditional=1 / 7,
                majority=sum(r["states"][j] == majority for r in dev) / len(dev),
                prior=sum(r["states"][j] == guess[r["start"]] for r in dev) / len(dev),
                prior_correct=sum(r["states"][j] == guess[r["start"]] for r in dev),
            )
        )
    return report


def build():
    if ROOT.exists():
        m = read(ROOT / "manifest.json")
        if m["generator"] != digest(Path(__file__)) or any(
            digest(ROOT / (k + ".jsonl")) != v for k, v in m["files"].items()
        ):
            raise ValueError("Pointer dataset identity changed; use a new namespace")
        return m
    parent = transitions.build()
    data = {}
    seen = set()
    for split in ("train", "development"):
        old = [
            json.loads(s)
            for s in (transitions.ROOT / (split + ".jsonl")).read_text().splitlines()
        ]
        data[split] = [instance(r["map"], r["start"]) for r in old]
        for r, o in zip(data[split], old, strict=True):
            assert (
                r["states"] == o["states"]
                and r["map_id"] == o["id"]
                and r["map_id"] not in seen
            )
            seen.add(r["map_id"])
    rng = random.Random(505109)
    reg = read(REG)
    confirm = []
    for _ in range(20000):
        m = rng.sample(range(8), 8)
        r = instance(m, rng.randrange(8))
        if r["map_id"] not in seen and cycle(r) > 3:
            confirm.append(r)
            seen.add(r["map_id"])
        if len(confirm) == reg["confirm_maps"]:
            break
    if len(confirm) != reg["confirm_maps"]:
        raise ValueError("Insufficient sealed maps")
    data["confirmation"] = confirm
    cases = []
    for r in data["development"]:
        m, s = r["map"], r["start"]
        path = [s, *r["states"][:2]]
        off = list(set(range(8)) - set(path))
        for _ in range(1000):
            perm = rng.sample(range(8), 8)
            new = [0] * 8
            for i in range(8):
                new[perm[i]] = perm[m[i]]
            rel = instance(new, perm[s])
            a, b = rng.sample(off, 2)
            changed = m.copy()
            changed[a], changed[b] = changed[b], changed[a]
            spare = instance(changed, s)
            a, b = path[1], rng.choice(off)
            changed = m.copy()
            changed[a], changed[b] = changed[b], changed[a]
            edit = instance(changed, s)
            if all(x["map_id"] not in seen for x in (rel, spare, edit)):
                break
        else:
            continue
        order = rng.sample(range(8), 8)
        if order == r["order"]:
            order.reverse()
        assert (
            spare["states"][:3] == r["states"][:3]
            and edit["states"][0] == r["states"][0]
            and edit["states"][1] != r["states"][1]
        )
        assert rel["states"] == [perm[x] for x in r["states"]]
        donor = instance(m, r["states"][0])
        cross = next(
            x
            for x in data["development"]
            if x["map_id"] != r["map_id"]
            and x["states"][0] != r["states"][0]
            and m[x["states"][0]] != x["states"][1]
        )
        same = next(
            instance(x["map"], x["map"].index(r["states"][0]))
            for x in data["development"]
            if x["map_id"] != r["map_id"]
        )
        cases.append(
            dict(
                original=r,
                order=instance(m, s, order),
                relabel=rel,
                perm=perm,
                off_path=spare,
                on_path=edit,
                other_start=donor,
                cross=cross,
                same_state=same,
            )
        )
        if len(cases) == reg["case_count"]:
            break
    if len(cases) != reg["case_count"]:
        raise ValueError("Insufficient counterfactual cases")
    data["cases"] = cases
    with TemporaryDirectory(dir=ROOT.parent) as td:
        dest = Path(td) / "dataset"
        dest.mkdir()
        files = {}
        for k, vals in data.items():
            p = dest / (k + ".jsonl")
            p.write_text(
                "".join(json.dumps(v, separators=(",", ":")) + "\n" for v in vals)
            )
            files[k] = digest(p)
        meta = dict(
            schema=5,
            license="MIT",
            generator=digest(Path(__file__)),
            parent=parent,
            files=files,
            prior=prior(data["train"], data["development"]),
            distributions={},
        )
        for k in ("train", "development", "confirmation"):
            rr = data[k]
            meta["distributions"][k] = dict(
                maps=len(rr),
                start_cycle=Counter(cycle(r) for r in rr),
                starts=Counter(r["start"] for r in rr),
                terminals=[Counter(r["states"][j] for r in rr) for j in range(6)],
            )
        save(dest / "manifest.json", meta)
        dest.rename(ROOT)
    return meta
