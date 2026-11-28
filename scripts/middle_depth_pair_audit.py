"""Independently reconstruct paired source scores, uncertainty and serving work."""

import argparse
import gzip
import json
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import product
from pathlib import Path
from statistics import mean

from loopqwen.io import digest, read, save

ROOT = Path("docs/measurements/middle-depth")
FAMILY_SIZES = {"chat": 3, "reasoning": 3, "tools": 3, "documents": 2}


def bounds(counts):
    # Inverse empirical CDF for all equiprobable stratified resamples.
    n = counts.total()
    cumulative = 0
    quantiles = []
    for point, mass in sorted(counts.items()):
        cumulative += mass
        for rank in (1, 39)[len(quantiles) :]:
            if cumulative * 40 >= n * rank:
                quantiles.append(float(point))
            else:
                break
    return quantiles


def source_uncertainty(groups):
    distribution = Counter({Fraction(0): 1})
    omitted = {}
    for family, sources in groups.items():
        values = list(sources.values())
        draws = Counter(
            mean(x) / len(groups) for x in product(values, repeat=len(values))
        )
        combined = Counter()
        for x, nx in distribution.items():
            for y, ny in draws.items():
                combined[x + y] += nx * ny
        distribution = combined
        for source in sources:
            scores = [
                mean(v for s, v in cluster.items() if f != family or s != source)
                for f, cluster in groups.items()
            ]
            omitted[f"{family}/{source}"] = float(mean(scores))
    return {
        "paired_sources": sum(map(len, groups.values())),
        "ordered_resamples": distribution.total(),
        "macro_difference": float(mean(mean(v.values()) for v in groups.values())),
        "exact_bootstrap_95": bounds(distribution),
        "macro_after_leaving_one_source_out": omitted,
    }


def serving(raw, arm):
    expected = [(layer, 0) for layer in range(24)]
    if arm == "loop":
        expected[16:16] = [(layer, 1) for layer in range(12, 16)]
    actual = raw.get("sites", raw.get("physical_sites"))
    if list(map(tuple, actual)) != expected:
        raise ValueError("Incorrect native layer execution order")
    attention = [site for site in expected if site[0] % 4 == 3]
    totals = Counter()
    for t in raw["trace"]["turns"]:
        prompt, generated = len(t["prompt_ids"]), len(t["token_ids"])
        reused = t.get("reused_prompt_tokens", 0)
        fresh = prompt - reused
        history = t["kv_history"]
        length = prompt + generated - 1
        if (
            not 0 < fresh <= prompt
            or generated <= 0
            or fresh != t.get("new_prompt_tokens", fresh)
            or [(h["layer"], h["turn"]) for h in history] != attention
            or any((h["retained"], h["offset"]) != (length, length) for h in history)
        ):
            raise ValueError(
                "Consumed positions disagree with retained native histories"
            )
        totals.update(
            model_turns=1,
            generated_tokens=generated,
            new_prompt_tokens=fresh,
            reused_prompt_tokens=reused,
            warm_layer_token_visits=(length - reused) * len(expected),
            logical_kv_file_bytes=t.get("logical_kv_file_bytes", 0),
        )
    return totals


def audit(label, cap):
    file = ROOT / "direct" / f"{label}-{cap}.json"
    result = read(file)["product"]
    groups = defaultdict(lambda: defaultdict(dict))
    costs = defaultdict(Counter)
    peaks = defaultdict(int)
    selected = read(ROOT / "direct-decision.json") if label == "chosen" else {}
    for r in result["rows"]:
        f, s, w = (r[k] for k in ("family", "source", "world"))
        if w in groups[f][s]:
            raise ValueError("Duplicate scored presentation")
        pair = {}
        openings = []
        for side, arm in (("left", "loop"), ("right", "base")):
            correct, seconds, name, sha, _ = r[side]
            path = ROOT / name
            if digest(path) != sha or type(correct) is not bool:
                raise ValueError(f"Unverified scored response {path}")
            with gzip.open(path) as stream:
                raw = json.load(stream)
            stage = "parent" if f == "documents" else "terminal"
            if arm == "loop" and selected:
                stage = selected[
                    "document_stage" if f == "documents" else "short_stage"
                ]
            review = "document-review" if f == "documents" else "review"
            locked = read(ROOT / review / stage / "scored.json")
            (match,) = [v for v in locked if v["arm"] == arm and v["file"] == name]
            if match["sha"] != sha or match["grade"]["correct"] is not correct:
                raise ValueError(f"Mismatch with locked review {name}")
            if correct and raw["trace"]["status"] != "answered":
                raise ValueError(f"Unfinished answer received credit {name}")
            pair[arm] = int(correct)
            turns = raw["trace"]["turns"]
            openings.append(turns[0]["prompt_ids"])
            key = f"{arm}/{'documents' if f == 'documents' else 'short'}"
            cost = costs[key]
            cost.update(serving(raw, arm))
            cost.update(
                requests=1,
                correct=int(correct),
                serving_seconds=raw["trace"]["seconds"],
            )
            cost["deadline_capped_time_to_correct_sum"] += (
                min(seconds, r["deadline"]) if correct else r["deadline"]
            )
            peaks[key] = max(peaks[key], *(t["peak_bytes"] for t in turns))
        if openings[0] != openings[1]:
            raise ValueError(f"Different actual paired prompts on {s}/{w}")
        groups[f][s][w] = Fraction(pair["loop"] - pair["base"])
    if {f: len(ss) for f, ss in groups.items()} != FAMILY_SIZES:
        raise ValueError("Incomplete source families")
    means = {}
    for f, sources in groups.items():
        worlds = (
            {"original", "edited"}
            if f == "documents"
            else {"original", "changed", "irrelevant"}
        )
        if any(set(by_world) != worlds for by_world in sources.values()):
            raise ValueError("Incomplete source enrollment")
        means[f] = {s: mean(ws.values()) for s, ws in sources.items()}
    stats = source_uncertainty(means)
    if abs(stats["macro_difference"] - result["macro_delta"]) > 1e-10:
        raise ValueError("Independent macro disagrees")
    for key, cost in costs.items():
        cost["peak_mlx_bytes"] = peaks[key]
        cost["mean_deadline_capped_time_to_correct"] = (
            cost.pop("deadline_capped_time_to_correct_sum") / cost["requests"]
        )
    return {
        "comparison_sha": digest(file),
        **stats,
        "registered_bootstrap_95": result["paired_interval"],
        "costs": dict(costs),
        "units": "Macro and bounds are proportions. Source clusters stay within families. The exact percentile bootstrap enumerates the empirical source distribution. With two or three sources per family its population coverage is uncertain. Warm layer-token visits count distinct layer/pass sites times fresh input and fed-back output positions. Logical KV bytes describe instrumented file accesses. Serving seconds exclude reusable prefix construction. Deadline-capped time uses cold book cost and charges failures their full deadline.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("label", choices=("parent", "chosen"))
    label = p.parse_args().label
    save(
        ROOT / "direct" / f"{label}-audit.json",
        {str(cap): audit(label, cap) for cap in (128, 512)},
    )
