"""Pair measured short and document outcomes with source weighting and equal deadlines."""

import argparse
from collections import defaultdict
from statistics import fmean

import numpy as np

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from scripts.middle_depth_budget import ROOT
from scripts.middle_depth_pairs import mechanism_item

FAMILIES = ("reasoning", "tools", "chat", "documents")


def observation(row, seconds):
    path = ROOT / row["file"]
    if digest(path) != row["sha"]:
        raise ValueError("Reviewed generation changed")
    raw = unpack(path)
    opening = fingerprint(raw["trace"]["turns"][0]["prompt_ids"])
    return row["grade"]["correct"], seconds, row["file"], row["sha"], opening


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
            out[key] = observation(r, r["seconds"])
    for r in read(ROOT / "document-review" / long_stage / "scored.json"):
        if (
            r["arm"] == arm
            and r["cap"] == cap
            and r["kind"] == "first"
            and r["world"] in ("original", "edited")
        ):
            key = "documents", r["source"], r["world"]
            out[key] = observation(r, r["seconds_cold"])
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
        left, right = a[key], b[key]
        if left[4] != right[4]:
            raise ValueError(f"Different actual opening tokens on {key}")
        deadline = 600 if f == "documents" else 8
        item = {
            "family": f,
            "source": s,
            "world": w,
            "left": left,
            "right": right,
            "deadline": deadline,
            "delta": int(left[0]) - int(right[0]),
            "deadline_delta": int(left[0] and left[1] <= deadline)
            - int(right[0] and right[1] <= deadline),
        }
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
        families[f] = {
            "left": an,
            "right": bn,
            "delta": an - bn,
            "deadline_delta": fmean(
                fmean(r["deadline_delta"] for r in rs) for rs in src.values()
            ),
            "gains": sum(r["delta"] == 1 for rs in src.values() for r in rs),
            "losses": sum(r["delta"] == -1 for rs in src.values() for r in rs),
            "sources": {s: fmean(r["delta"] for r in rs) for s, rs in src.items()},
        }
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
    curves = {}
    for f, src in grouped.items():
        deadlines = (
            (2, 4, 8, 16, 32, 64)
            if f != "documents"
            else (150, 300, 450, 600, 900, 1200)
        )
        curves[f] = {
            str(deadline): {
                side: fmean(
                    fmean(r[side][0] and r[side][1] <= deadline for r in rs)
                    for rs in src.values()
                )
                for side in ("left", "right")
            }
            for deadline in deadlines
        }
    hard = defaultdict(list)
    for r in rows:
        if mechanism_item(r["family"], r["world"]):
            hard[r["source"]].append(r["deadline_delta"])
    hard_effects = {s: fmean(values) for s, values in hard.items()}
    return {
        "rows": rows,
        "quality_by_deadline": curves,
        "registered_short_mechanism": {
            "source_effects": hard_effects,
            "matched_eight_second_delta": fmean(hard_effects.values()),
            "reaches_five_points": fmean(hard_effects.values()) >= 0.05,
        },
        "families": families,
        "macro_delta": macro,
        "paired_interval": interval,
        "deadline_macro_delta": fmean(r["deadline_delta"] for r in families.values()),
        "changed_pairs": {
            f: {
                source: {
                    side: all(
                        r[side][0]
                        for r in rs
                        if r["world"] in ("original", "changed", "edited")
                    )
                    for side in ("left", "right")
                }
                for source, rs in src.items()
            }
            for f, src in grouped.items()
        },
        "prior_successes_lost": sum(r["delta"] == -1 for r in rows),
        "prior_successes": sum(r["right"][0] for r in rows),
        "book_bands": {
            s: {
                "left": fmean(r["left"][0] for r in rs),
                "right": fmean(r["right"][0] for r in rs),
            }
            for s, rs in grouped["documents"].items()
        },
        "scope": "Development clusters only. Generalization and replicated causal depth require separate enrollment.",
    }


def product_gates(product):
    minimum = (
        product["macro_delta"] >= 0.05
        and product["paired_interval"][0] > 0
        and all(
            v["delta"] >= 0 and v["one_sided_simultaneous_lower"] > -0.05
            for v in product["families"].values()
        )
    )
    strong = (
        minimum
        and product["macro_delta"] >= 0.10
        and all(v["delta"] > 0 for v in product["families"].values())
        and product["deadline_macro_delta"] > 0
    )
    books = (
        all(v["left"] >= 0.50 for v in product["book_bands"].values())
        and product["families"]["documents"]["delta"] > 0
    )
    return {"minimum_product": minimum, "strong_product": strong, "long_context": books}


def measure(a):
    candidate = panel("loop", a.candidate_short, a.candidate_long, a.cap)
    ordinary = panel("ordinary", a.ordinary_short, a.ordinary_long, a.cap)
    base = panel("base", a.base_short, a.base_long, a.cap)
    product = contrast(candidate, base)
    depth = contrast(candidate, ordinary)
    save(
        ROOT / "four-families" / (a.name + ".json"),
        {
            "cap": a.cap,
            "product": product,
            "matched_depth": depth,
            "development_thresholds": product_gates(product),
            "positive_development": product["macro_delta"] > 0,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--cap", type=int, choices=(128, 512), required=True)
    for role in ("candidate", "ordinary", "base"):
        for part in ("short", "long"):
            p.add_argument("--" + role + "-" + part, required=True)
    measure(p.parse_args())
