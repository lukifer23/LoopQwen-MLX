"""Score unrestricted next tokens, categorical retention and source-disjoint prose."""

import gc
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import mlx.core as mx
from ..data import retention3, pointer5, pointer6
from ..identity import fingerprint
from ..io import read, save, digest
from ..training.transfer7 import (
    ARM_T,
    T,
    P,
    OUT,
    ARM,
    example,
    digit_ids,
    vocab,
    kernel,
)
from ..training.losses import validation_nll
from .mechanism import load, packed, unpack, evaluate
from .order_scores import metrics
from .language7 import base

DEST = OUT / "transfer"


def heldout(m, t, weights, name, base_model=False):
    digits = digit_ids(t)
    rows = [
        example(pointer6.ordered(r, "random0", 61091), "state machine", t, digits)
        for r in pointer5.rows("development")[: T["evaluation_maps"]]
    ]
    identity = fingerprint(
        [weights, [r["ids"] for r in rows], T, digest(Path(__file__)), P]
    )
    path = DEST / (name + ".json.gz")
    if path.exists():
        old = unpack(path)
        if identity != old["identity"]:
            raise ValueError("Incompatible digit evaluation")
        return old["summary"]
    records = []
    start = perf_counter()
    for i in range(0, len(rows), 8):
        group = rows[i : i + 8]
        x, last = kernel.inputs(group)
        logits = []
        states = []
        for h in m.block_states(x, 1 if base_model else 3):
            z = vocab(m, h, last)
            logits.append(mx.argmax(z, axis=-1).tolist())
            if not base_model:
                states.append(kernel.readout(m, h, last).argmax(-1).tolist())
        for j, q in enumerate(group):
            gold = q["row"]["states"][:3]
            records.append(
                dict(
                    id=q["row"]["example_id"],
                    map=q["row"]["map"],
                    start=q["row"]["start"],
                    ids=q["ids"],
                    order=q["row"]["order"],
                    gold=gold,
                    digit_targets=[digits[v] for v in gold],
                    predicted_ids=[v[j] for v in logits],
                    readout=[v[j] for v in states],
                )
            )

    def counts(key, labels):
        return [
            sum(r[key][i] == r[labels][i] for r in records)
            for i in range(len(records[0][key]))
        ]

    summary = dict(
        n=len(records),
        vocab=counts("predicted_ids", "digit_targets"),
        state=counts("readout", "gold"),
        first_exit_as_third=sum(
            r["predicted_ids"][0] == r["digit_targets"][2] for r in records
        ),
        seconds=perf_counter() - start,
    )
    packed(
        path, dict(identity=identity, weights=weights, summary=summary, records=records)
    )
    return summary


def retention(m, weights, name, k):
    manifest = retention3.build()
    rows = [
        json.loads(l)
        for l in (retention3.ROOT / "validation.jsonl").read_text().splitlines()
    ]
    m.span = replace(m.span, passes=k)
    by = {}
    for domain in sorted({q["category"] for q in rows}):
        by[domain] = validation_nll(
            m,
            [q for q in rows if q["category"] == domain],
            batch_size=2,
            projection_chunk=32,
        )
    n = sum(v["tokens"] for v in by.values())
    return dict(
        weights=weights,
        depth=k,
        corpus=manifest["sha256"],
        passages=len(rows),
        tokens=n,
        by_domain=by,
        nll=sum(v["nll"] * v["tokens"] for v in by.values()) / n,
    )


def assess(st):
    DEST.mkdir(parents=True, exist_ok=True)
    final = st["step"]
    dec = read(OUT / "decision.json")
    sets = [("base", None, 0), ("installed", ARM, dec["step"])]
    sets += [
        (f"transfer-{n}", ARM_T, n)
        for n in sorted(
            set([n for n in T["milestones"] if n <= final] + ([final] if final else []))
        )
    ]
    stats = {}
    retained = {}
    for name, arm, n in sets:
        m, t, weights = (
            base() if arm is None else (lambda x: (x[0], x[1], x[3]))(load(arm, n))
        )
        stats[name] = heldout(m, t, weights, name, arm is None)
        print(name, stats[name], flush=True)
        if name in ("base", "installed", f"transfer-{final}"):
            for k in [1] if arm is None else [1, 3]:
                retained[f"{name}/K{k}"] = retention(m, weights, name, k)
        del m, t
        gc.collect()
        mx.clear_cache()
    result = dict(step=final, status=st["status"], verbal=stats, retention=retained)
    if final:
        data = evaluate(
            ARM_T,
            final,
            pointer5.rows("development"),
            pointer6.SPEC["orders"],
            3,
            DEST / "orders.json.gz",
        )
        c = metrics(data["records"])["conditions"]["mean_random"]["accuracy"][-1]
        gain = (
            stats[f"transfer-{final}"]["vocab"][-1] - stats["installed"]["vocab"][-1]
        ) / T["evaluation_maps"]
        result.update(
            categorical=c,
            gain=gain,
            checks=dict(
                gain=gain >= 0.03,
                categorical=c
                >= dec["scores"]["conditions"]["mean_random"]["accuracy"][-1]
                - T["categorical_max_drop"],
                retention=retained[f"transfer-{final}/K1"]["nll"]
                <= min(retained["base/K1"]["nll"], retained["installed/K1"]["nll"])
                + T["retention_k1_max_increase"],
            ),
        )
    save(DEST / "decision.json", result)
