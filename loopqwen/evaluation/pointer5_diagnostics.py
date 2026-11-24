"""Behavioral and causal development tests at immutable pointer milestones."""

from pathlib import Path
from time import perf_counter

import mlx.core as mx

from .. import checkpoint
from ..data.pointer5 import rows, ROOT
from ..io import digest, read, save
from ..training.pointer5 import (
    OUT,
    architecture,
    install,
    root_for,
    inputs,
    request_rows,
    readout,
    reenter,
)
from .pointer5 import milestone, installed, evaluate


def load(seed, arm, step):
    measured = milestone(seed, arm, step)
    root = root_for(seed, arm)
    gen = root / read(root / f"at-{step}.json")["directory"]
    m, tok, _, _ = architecture(seed, arm, read(OUT / "profile.json")["selected"])
    checkpoint.restore(gen, m)
    m.eval()
    return m, tok, measured["weights"]


def behavioral(m, tok, arm, cases):
    records = {}
    for kind in ("original", "relabel", "order", "on_path", "off_path", "other_start"):
        measures = evaluate(m, tok, arm, [c[kind] for c in cases])
        records[kind] = measures
    for kind, test in records.items():
        for j, metric in enumerate(test["metrics"]):
            agree = both = changed_gold = correct_change = 0
            for c, orig, row in zip(
                cases, records["original"]["records"], test["records"], strict=True
            ):
                a, b = orig["pred"][j], row["pred"][j]
                if kind == "relabel":
                    a = c["perm"][a]
                agree += a == b
                both += orig["pred"][j] == orig["gold"][j] and b == row["gold"][j]
                if orig["gold"][j] != row["gold"][j]:
                    changed_gold += 1
                    correct_change += b == row["gold"][j] and b != orig["pred"][j]
            metric.update(
                prediction_agreement=agree,
                both_correct=both,
                changed_gold=changed_gold,
                correct_changed_answers=correct_change,
            )
    return records


def transplant(m, tok, arm, cases, seed):
    """Only hidden vectors enter this patch; labels are inspected after prediction."""
    output = []
    for i in range(0, len(cases), 8):
        group = cases[i : i + 8]
        cache = {}

        def first(kind):
            r = request_rows([c[kind] for c in group], 1, arm, tok, cache)
            x, last = inputs(r)
            p = m.prefix_hidden(x)
            h = m.shared_block(p)
            return p, h, last

        p, h, last = first("original")
        batch = mx.arange(len(group))
        v = h[batch, last]
        base_pred = mx.argmax(readout(m, h, last), axis=-1).tolist()

        def next_prediction(carried):
            pred = m.shared_block(reenter(m, carried, p, arm))
            return mx.argmax(readout(m, pred, last), axis=-1).tolist()

        no_patch = next_prediction(h)
        for kind in ("other_start", "cross", "same_state"):
            _, donor, last_d = first(kind)
            d = donor[batch, last_d]
            donor_pred = mx.argmax(readout(m, donor, last_d), axis=-1).tolist()
            # Exact replacement rather than BF16 subtract/add roundoff.
            patched = mx.where(
                (mx.arange(h.shape[1])[None, :] == last[:, None])[:, :, None],
                d[:, None, :],
                h,
            )
            pred = next_prediction(patched)
            delta = d.astype(mx.float32) - v.astype(mx.float32)
            noise = mx.random.normal(delta.shape, key=mx.random.key(seed + i))
            perturb = (
                noise
                * mx.linalg.norm(delta, axis=-1, keepdims=True)
                / mx.maximum(mx.linalg.norm(noise, axis=-1, keepdims=True), 1e-8)
            )
            rand = mx.where(
                (mx.arange(h.shape[1])[None, :] == last[:, None])[:, :, None],
                (v + perturb).astype(h.dtype)[:, None, :],
                h,
            )
            control = next_prediction(rand)
            for j, c in enumerate(group):
                r = c["original"]
                donor_record = c[kind]
                gold = donor_record["states"][0]
                output.append(
                    dict(
                        kind=kind,
                        recipient=r,
                        donor=donor_record,
                        donor_read=donor_pred[j],
                        recipient_read=base_pred[j],
                        expected=r["map"][gold],
                        donor_map_next=donor_record["map"][gold],
                        recipient_next=r["states"][1],
                        natural=no_patch[j],
                        patch=pred[j],
                        noise=control[j],
                    )
                )
    summary = {}
    for kind in ("other_start", "cross", "same_state"):
        all_rows = [r for r in output if r["kind"] == kind]
        for label, selected in (
            ("all", all_rows),
            (
                "correct_donor",
                [r for r in all_rows if r["donor_read"] == r["donor"]["states"][0]],
            ),
        ):
            summary[kind + "/" + label] = dict(
                n=len(selected),
                donor_transition=sum(r["patch"] == r["expected"] for r in selected),
                natural_donor_transition=sum(
                    r["natural"] == r["expected"] for r in selected
                ),
                noise_donor_transition=sum(
                    r["noise"] == r["expected"] for r in selected
                ),
                recipient_persistence=sum(
                    r["patch"] == r["recipient_next"] for r in selected
                ),
            )
    return dict(records=output, summary=summary)


def diagnose(seed, arm, step=480):
    m, tok, weights = load(seed, arm, step)
    cases = rows("cases")
    begin = perf_counter()
    data = behavioral(m, tok, arm, cases)
    patches = transplant(m, tok, arm, cases, seed)
    # Inspect amplification of the carried state across all prompt positions.
    req = request_rows([c["original"] for c in cases[:8]], 3, arm, tok, {})
    x, last = inputs(req)
    p = m.prefix_hidden(x)
    h = p
    norms = []
    for j in range(3):
        u = reenter(m, h, p, arm) if j else h
        nxt = m.shared_block(u)
        rms = lambda v: mx.sqrt(mx.mean(v.astype(mx.float32).square())).item()
        norms.append(
            dict(
                input_rms=rms(u),
                output_rms=rms(nxt),
                bridge_rms=rms(u - h),
                change_rms=rms(nxt - h),
            )
        )
        h = nxt
    save(
        OUT / f"{seed}/diagnostics-{arm}-{step}.json",
        dict(
            weights=weights,
            cases=digest(ROOT / "cases.jsonl"),
            implementation=digest(Path(__file__)),
            behavior=data,
            intervention=patches,
            norms=norms,
            seconds=perf_counter() - begin,
        ),
    )
    return patches["summary"]


def followup(seed):
    choice = "recurrent"
    if not installed(seed):
        install(seed, "bridge", 480)
        diagnose(seed, "bridge")
        choice = "bridge"
    if not installed(seed, choice):
        print(
            "Installation criterion missed at trained depths. Closed extrapolation and annealing.",
            flush=True,
        )
        return
    for arm in ("multi", "terminal"):
        install(seed, arm, 480)
    for arm in (choice, "middle", "full"):
        m, tok, weights = load(seed, arm, 480)
        result = evaluate(m, tok, arm, rows("development"), (1, 2, 3, 4, 5, 6))
        save(OUT / f"{seed}/extra-{arm}.json", dict(weights=weights, **result))
    if choice == "recurrent":
        install(seed, "outcome", 560)
        diagnose(seed, "outcome", 560)
        m, tok, weights = load(seed, "outcome", 560)
        save(
            OUT / f"{seed}/extra-outcome.json",
            dict(
                weights=weights,
                **evaluate(m, tok, "outcome", rows("development"), (1, 2, 3, 4, 5, 6)),
            ),
        )
