"""Pair measured short and document outcomes with source weighting and equal deadlines."""

import argparse
from collections import defaultdict
from statistics import fmean

import numpy as np

from loopqwen.io import read, save
from scripts.middle_depth_budget import ROOT

FAMILIES = ("reasoning", "tools", "chat", "documents")


def panel(arm, short_stage, long_stage, cap):
    out = {}
    short = read(ROOT / "review" / short_stage / "scored.json")
    for r in short:
        if (
            r["arm"] == arm
            and r["cap"] == cap
            and r["world"] in ("original", "changed", "irrelevant")
        ):
            key = r["family"], r["source"], r["world"]
            out[key] = (r["grade"]["correct"], r["seconds"], r["file"], r["sha"])
    for r in read(ROOT / "document-review" / long_stage / "scored.json"):
        if r["arm"] == arm and r["cap"] == cap and r["kind"] == "first":
            key = "documents", r["source"], r["world"]
            out[key] = (r["grade"]["correct"], r["seconds_cold"], r["file"], r["sha"])
    if len(out) != 31:
        raise ValueError(
            f"Expected 27 short and four book presentations for {arm}, found {len(out)}"
        )
    return out


def contrast(a, b):
    if a.keys() != b.keys():
        raise ValueError("Incomplete matched enrollment")
    grouped = defaultdict(lambda: defaultdict(list))
    rows = []
    for key in sorted(a):
        f, s, w = key
        l, r = a[key], b[key]
        deadline = 600 if f == "documents" else 8
        item = dict(
            family=f,
            source=s,
            world=w,
            left=l,
            right=r,
            deadline=deadline,
            delta=int(l[0]) - int(r[0]),
            deadline_delta=int(l[0] and l[1] <= deadline)
            - int(r[0] and r[1] <= deadline),
        )
        grouped[f][s].append(item)
        rows.append(item)
    families = {}
    samples = []
    for f in FAMILIES:
        src = grouped[f]
        assert len(src) == (2 if f == "documents" else 3)
        an = fmean(fmean(r["left"][0] for r in rs) for rs in src.values())
        bn = fmean(fmean(r["right"][0] for r in rs) for rs in src.values())
        gains = [fmean(r["delta"] for r in rs) for rs in src.values()]
        samples.append(gains)
        families[f] = dict(
            left=an,
            right=bn,
            delta=an - bn,
            deadline_delta=fmean(
                fmean(r["deadline_delta"] for r in rs) for rs in src.values()
            ),
            gains=sum(r["delta"] == 1 for rs in src.values() for r in rs),
            losses=sum(r["delta"] == -1 for rs in src.values() for r in rs),
            sources={s: fmean(r["delta"] for r in rs) for s, rs in src.items()},
        )
    # Repeated presentations of one source remain in the same bootstrap unit.
    rng = np.random.default_rng(4619)
    draws = [
        np.asarray(v)[rng.integers(0, len(v), size=(20000, len(v)))].mean(axis=1)
        for v in samples
    ]
    for f, x in zip(FAMILIES, draws, strict=True):
        families[f]["one_sided_simultaneous_lower"] = float(
            np.quantile(x, 0.05 / len(FAMILIES))
        )
    macro = fmean(r["delta"] for r in families.values())
    interval = np.quantile(np.mean(draws, axis=0), [0.025, 0.975]).tolist()
    return dict(
        rows=rows,
        families=families,
        macro_delta=macro,
        paired_interval=interval,
        deadline_macro_delta=fmean(r["deadline_delta"] for r in families.values()),
        prior_successes=sum(r["right"][0] for r in rows),
        book_bands={
            s: dict(
                left=fmean(r["left"][0] for r in rs),
                right=fmean(r["right"][0] for r in rs),
            )
            for s, rs in grouped["documents"].items()
        },
        scope="Development clusters only. Generalization and replicated causal depth require separate enrollment.",
    )


def measure(a):
    candidate = panel("loop", a.candidate_short, a.candidate_long, a.cap)
    ordinary = panel("ordinary", a.ordinary_short, a.ordinary_long, a.cap)
    base = panel("base", a.base_short, a.base_long, a.cap)
    product = contrast(candidate, base)
    depth = contrast(candidate, ordinary)
    save(
        ROOT / "four-families" / (a.name + ".json"),
        dict(
            cap=a.cap,
            product=product,
            matched_depth=depth,
            positive_development=product["macro_delta"] > 0,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--cap", type=int, choices=(128, 512), required=True)
    for role in ("candidate", "ordinary", "base"):
        for part in ("short", "long"):
            p.add_argument("--" + role + "-" + part, required=True)
    measure(p.parse_args())
