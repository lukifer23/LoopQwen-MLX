"""Map-clustered accuracy, paired differences, calibration and independent orbit scoring."""

import math
from collections import Counter, defaultdict

import numpy as np


def orbit(mapping, start, depth):
    if not 0 <= start < len(mapping) or any(
        x not in range(len(mapping)) for x in mapping
    ):
        raise ValueError("Invalid functional graph")
    path = []
    node = start
    while node not in path:
        path.append(node)
        node = mapping[node]
    cut = path.index(node)
    loop = path[cut:]

    def at(i):
        return path[i] if i < len(path) else loop[(i - cut) % len(loop)]

    return [at(i) for i in range(1, depth + 1)], cut, len(loop)


def parse(records):
    by = defaultdict(dict)
    for r in records:
        order = r["order"]
        if sorted(order) != list(range(len(r["map"]))):
            raise ValueError("Invalid edge presentation")
        gold, _, _ = orbit(r["map"], r["start"], len(r["gold"]))
        p = np.array(r["probabilities"])
        if (
            gold != r["gold"]
            or p.shape != (len(gold), len(r["map"]))
            or not np.isfinite(p).all()
            or (p < 0).any()
            or np.max(np.abs(p.sum(axis=-1) - 1)) > 1e-5
        ):
            raise ValueError("Invalid orbit or recorded probabilities")
        key = (tuple(r["map"]), r["start"], r["example_id"])
        group = by[r["condition"]]
        if key in group:
            raise ValueError("Duplicate map/start within order condition")
        group[key] = (p, tuple(order), gold)
    keys = list(next(iter(by.values())))
    if any(list(group) != keys for group in by.values()):
        raise ValueError("Unpaired map cohorts")
    return keys, by


def metrics(records):
    keys, by = parse(records)
    order = list(by)
    first = order.index("canonical")
    p = np.array([[by[k][key][0] for k in order] for key in keys])
    y = np.array([by[order[0]][key][2] for key in keys])[:, None, :]
    pred = p.argmax(axis=-1)
    hit = pred == y
    trajectory = hit.cumprod(axis=-1)
    same = pred == pred[:, first : first + 1, :]
    both = hit & hit[:, first : first + 1, :]
    nll = -np.log(
        np.maximum(
            1e-38,
            np.take_along_axis(p, np.broadcast_to(y, hit.shape)[..., None], axis=-1)[
                ..., 0
            ],
        )
    )
    conf = p.max(axis=-1)
    rows = {}
    groups = {k: [i] for i, k in enumerate(order)}
    randoms = [i for i, k in enumerate(order) if k.startswith("random")]
    if randoms:
        groups["mean_random"] = randoms
    for k, ii in groups.items():
        h = hit[:, ii]
        cf = conf[:, ii]
        ece = []
        for j in range(h.shape[-1]):
            e = 0
            for low in range(10):
                mask = np.minimum(9, (cf[:, :, j] * 10).astype(int)) == low
                if mask.any():
                    e += mask.mean() * abs(
                        h[:, :, j][mask].mean() - cf[:, :, j][mask].mean()
                    )
            ece.append(float(e))
        rows[k] = dict(
            accuracy=h.mean(axis=(0, 1)).tolist(),
            trajectory=trajectory[:, ii].mean(axis=(0, 1)).tolist(),
            consistent=same[:, ii].mean(axis=(0, 1)).tolist(),
            both_correct=both[:, ii].mean(axis=(0, 1)).tolist(),
            nll=nll[:, ii].mean(axis=(0, 1)).tolist(),
            ece=ece,
        )
    indices = np.random.default_rng(61091).integers(
        0, len(keys), size=(10000, len(keys))
    )
    hm = hit[:, randoms].mean(axis=1) if randoms else hit[:, first]
    tm = (
        trajectory[:, randoms, -1].mean(axis=1) if randoms else trajectory[:, first, -1]
    )
    ci = lambda v: np.quantile(v[indices].mean(axis=1), [0.025, 0.975], axis=0).tolist()
    return dict(
        conditions=rows,
        n=len(keys),
        worst_order=hit.mean(axis=0).min(axis=0).tolist(),
        correct_every_order=hit.all(axis=1).mean(axis=0).tolist(),
        accuracy_ci=ci(hm),
        trajectory_ci=ci(tm),
        per_map=[
            dict(id=key[2], random_accuracy=h.tolist(), random_complete=float(t))
            for key, h, t in zip(keys, hm, tm, strict=True)
        ],
    )


def paired(a, b):
    left = a["per_map"]
    right = b["per_map"]
    if [r["id"] for r in left] != [r["id"] for r in right]:
        raise ValueError("Unpaired IDs")
    d = np.array([r["random_accuracy"] for r in right]) - np.array(
        [r["random_accuracy"] for r in left]
    )
    indices = np.random.default_rng(61091).integers(0, len(d), size=(10000, len(d)))
    return dict(
        difference=d.mean(axis=0).tolist(),
        interval=np.quantile(d[indices].mean(axis=1), [0.025, 0.975], axis=0).tolist(),
    )


def deeper(records):
    bins = defaultdict(Counter)
    signals = defaultdict(list)
    for r in records:
        pred = np.argmax(r["probabilities"], axis=-1)
        gold, transient, cycle = orbit(r["map"], r["start"], len(pred))
        history = [r["start"]]
        for j, (s, g, activ) in enumerate(
            zip(pred, gold, r["activations"], strict=True)
        ):
            wraps = j + 1 >= transient + cycle
            k = f"{r['condition']}/{j + 1}/{'wrapped' if wraps else 'new'}"
            b = bins[k]
            b["n"] += 1
            b["correct"] += int(s == g)
            b["trajectory"] += int(list(pred[: j + 1]) == gold[: j + 1])
            b["follows_predicted"] += int(s == r["map"][history[-1]])
            b["stays"] += int(s == history[-1])
            b["fails_to_advance"] += int(
                s == history[-1] and r["map"][history[-1]] != s
            )
            b["early_return"] += int(s in history and g not in [r["start"], *gold[:j]])
            b["correct_cycle_return"] += int(g == r["start"] and s == g)
            b["correct_probability_sum"] += r["probabilities"][j][g]
            signals[k + ("/correct" if s == g else "/wrong")].append(activ)
            history.append(s)
    return dict(
        groups=dict(bins),
        activation_means={k: np.mean(v, axis=0).tolist() for k, v in signals.items()},
        activation_columns=[
            "readout_rms",
            "readout_update_rms",
            "cosine",
            "context_rms",
            "context_update_rms",
        ],
    )
