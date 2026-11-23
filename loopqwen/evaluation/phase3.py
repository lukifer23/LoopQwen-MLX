"""Development-only, milestone-addressed comparisons for the Phase 3 study."""

import argparse
import gc
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np

from ..checkpoint import load_trained
from ..data.v2 import rows
from ..identity import fingerprint
from .identity import evaluation_identity
from ..io import read, save
from ..training.losses import validation_nll
from .phase2 import predictions
from .statistics import exact_mcnemar
from .trace_verifier import verify

OUT = Path("outputs/phase3")
REGISTER = Path("experiments/phase3_dev.json")


def token_classes(row, tok):
    response = " " + row["trace"]
    encoded = tok._tokenizer(
        response, add_special_tokens=False, return_offsets_mapping=True
    )
    if encoded["input_ids"] + [tok.eos_token_id] != row["ids"][row["prefix"] :]:
        raise ValueError("Teacher-forced response tokenization changed")
    boundary = response.rindex("Final answer:") + len("Final answer:")
    result = []
    for begin, end in encoded["offset_mapping"]:
        chars = response[begin:end]
        if end > boundary and response[max(begin, boundary) : end].strip():
            group = "final"
        elif begin < boundary and re.search(r"[0-9+*/%\-]", chars):
            group = "intermediate"
        else:
            group = "format"
        result.append(group)
    return result + ["eos"]


def likelihood(model, tok, samples):
    sums = Counter()
    counts = Counter()
    details = []
    for row in samples:
        ids = row["ids"]
        logits_h = model.hidden(mx.array([ids[:-1]]))[0, row["prefix"] - 1 :]
        y = mx.array(ids[row["prefix"] :])
        groups = token_classes(row, tok)
        losses = []
        entropies = []
        for i in range(0, len(groups), 24):
            logits = model.project(logits_h[i : i + 24]).astype(mx.float32)
            logz = mx.logsumexp(logits, axis=-1)
            entropy = logz - (mx.softmax(logits, axis=-1) * logits).sum(-1)
            loss = mx.fast.cross_entropy(logits, y[i : i + 24])
            mx.eval(loss, entropy)
            losses.extend(loss.tolist())
            entropies.extend(entropy.tolist())
        final_logp = 0.0
        for group, loss in zip(groups, losses, strict=True):
            counts[group] += 1
            sums[group] += loss
            if group == "final":
                final_logp -= loss
        details.append(
            dict(
                id=row["id"],
                final_logp=final_logp,
                entropy=sum(entropies) / len(entropies),
            )
        )
    return dict(
        tokens=dict(counts),
        nll={k: s / counts[k] for k, s in sums.items()},
        trace_nll=sum(sums.values()) / sum(counts.values()),
        records=details,
    )


def compare(left, right):
    a = left["generations"]["records"]
    b = right["generations"]["records"]
    if [r["id"] for r in a] != [r["id"] for r in b]:
        raise ValueError("Pairing requires the same ordered IDs")
    change = np.asarray([int(y["correct"]) - int(x["correct"]) for x, y in zip(a, b)])
    wins = int(sum(change > 0))
    losses = int(sum(change < 0))
    ci = np.quantile(
        np.random.default_rng(73).choice(change, (10000, len(a))).mean(axis=1),
        [0.025, 0.975],
    )
    return dict(
        fixes=wins,
        breaks=losses,
        delta=float(change.mean()),
        item_ci95=ci.tolist(),
        mcnemar=exact_mcnemar(wins, losses),
        same_numeric=sum(
            x["prediction"] == y["prediction"] and x["prediction"] is not None
            for x, y in zip(a, b)
        ),
        formatting_only=sum(
            x["text"] != y["text"]
            and "".join(x["text"].split()) == "".join(y["text"].split())
            for x, y in zip(a, b)
        ),
        unchanged_text_improved_logp=sum(
            x["text"] == y["text"] and q["final_logp"] > p["final_logp"]
            for x, y, p, q in zip(
                a, b, left["likelihood"]["records"], right["likelihood"]["records"]
            )
        ),
    )


def measure(arm, milestone, mode, samples, prose, study):
    root = Path("checkpoints") / ("p3_" + arm)
    gen = root / read(root / f"targets-{milestone}.json")["directory"]
    setting = study["generation"]
    identity = evaluation_identity(
        study,
        gen,
        [r["id"] for r in samples + prose],
        {**setting, "mode": mode, "batch": 8, "left_padded": True},
        "procedural-certificate-v1",
    )
    dest = (
        OUT / "eval" / str(milestone) / arm / mode / (fingerprint(identity) + ".json")
    )
    if dest.exists():
        previous = read(dest)
        if previous["identity"] != identity:
            raise ValueError("Evaluation identity mismatch")
        return previous
    model, tok, state = load_trained(gen)
    if mode == "bypass":
        model.span = replace(model.span, passes=1)
    elif mode.startswith("alpha"):
        model.alpha_override = float(mode[5:])
    mx.eval(model.hidden(mx.array([[1, 2, 3]])))
    result = predictions(
        model, tok, samples, setting["cap"], padded=True, detailed=True
    )
    counts = Counter()
    for pred, row in zip(result["records"], samples):
        certificate = verify(row, pred["text"], pred["token_cap_reached"])
        pred["verification"] = certificate
        pred["correct"] = certificate["final"] == "correct"
        for key in (
            "verified",
            "format_compliant",
            "truncated",
            "recognized",
            "correct_steps",
            "expected_steps",
        ):
            counts[key] += certificate[key]
        counts["emitted_steps"] += len(certificate["checks"])
        counts["correct"] += pred["correct"]
    result["correct"] = counts["correct"]
    cert = dict(
        counts,
        verification_coverage=counts["recognized"] / max(1, counts["emitted_steps"]),
        conditional_step_accuracy=counts["correct_steps"]
        / max(1, counts["recognized"]),
    )
    record = dict(
        identity=identity,
        arm=arm,
        mode=mode,
        milestone=milestone,
        order=state["order_hash"],
        input_tokens=state["input_tokens"],
        targets=state["loss_tokens"],
        generation=gen.name,
        alpha=float(model.effective_alpha())
        if model.span.has_gate and model.span.passes > 1
        else None,
        executions=len(model.sites),
        certificate=cert,
        generations=result,
        likelihood=likelihood(model, tok, samples),
        retention=validation_nll(model, prose),
    )
    save(dest, record)
    print(
        f"{milestone} {arm} {mode}: {counts['correct']}/{len(samples)}, {counts['verified']} verified",
        flush=True,
    )
    del model
    gc.collect()
    mx.clear_cache()
    return record


def evaluate(stage, candidate):
    study = read(REGISTER)
    screen = stage == "screen"
    target = 25000 if screen else 100000
    arms = (
        ["dense", "direct", "fixed", candidate]
        if screen
        else ["dense", "direct", candidate]
    )
    setting = study["generation"]
    samples = rows(setting["split"], "trace")[: setting["count"]]
    prose = rows("language_validation", "language")
    score = {}
    pairs = {}
    table = []
    for arm in arms:
        modes = ["trained"]
        if arm != "dense":
            modes += ["bypass"]
        if arm not in ("dense", "direct") and not screen:
            modes += ["alpha0", "alpha0.05", "alpha0.10"]
        for mode in modes:
            r = measure(arm, target, mode, samples, prose, study)
            score[arm, mode] = r
            table.append(
                dict(
                    arm=arm,
                    mode=mode,
                    correct=r["certificate"]["correct"],
                    verified=r["certificate"]["verified"],
                    truncated=r["certificate"]["truncated"],
                    n=len(samples),
                    alpha=r["alpha"],
                    final_nll=r["likelihood"]["nll"]["final"],
                    trace_nll=r["likelihood"]["trace_nll"],
                    retention=r["retention"]["nll"],
                    seconds=r["generations"]["seconds"],
                    identity=fingerprint(r["identity"]),
                )
            )
        if arm != "dense":
            pairs[arm + "-dense"] = compare(
                score["dense", "trained"], score[arm, "trained"]
            )
            pairs[arm + "-bypass"] = compare(
                score[arm, "bypass"], score[arm, "trained"]
            )
    if (
        len({(r["order"], r["targets"], r["input_tokens"]) for r in score.values()})
        != 1
    ):
        raise ValueError("Unmatched training exposure")
    summary = dict(stage=stage, milestone=target, rows=table, paired=pairs)
    if screen:
        best = max(
            (
                r
                for r in table
                if r["mode"] == "trained" and r["arm"] in ("fixed", candidate)
            ),
            key=lambda r: (r["correct"], r["verified"], -r["final_nll"]),
        )
        save(OUT / "selection.json", dict(selected=best["arm"], evidence=table))
    save(OUT / (stage + ".json"), summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["screen", "main"])
    parser.add_argument("candidate", choices=["fixed", "learn1", "learn2"])
    a = parser.parse_args()
    evaluate(a.stage, a.candidate)
