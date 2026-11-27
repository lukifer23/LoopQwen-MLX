"""Independently reproduce the supervised loss from public cached logits and validate update exposure."""

from collections import Counter
from pathlib import Path
from statistics import mean
from time import monotonic

import mlx.core as mx

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import read, digest
from loopqwen.shared_query_assistant import SharedQueryAssistant
from scripts.receipt_repair_fit import OUT as FIT, PLAN, PARENT, EPOCHS
from scripts.receipt_repair_material import MATERIAL
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

OUT = Path("docs/measurements/receipt-repair-cold.json")


def run():
    committed(FIT / "receipt.json")
    fit = read(FIT / "receipt.json")
    if "after" not in fit:
        return
    assert fit["plan"] == digest(PLAN)
    policy = FIT / "research-policy"
    model = SharedQueryAssistant(
        read(PARENT / "registration.json")["checkpoint"], policy
    )._tool_model
    assert model.revisits == 2 and digest(policy / "step.safetensors") == fit["weights"]
    rows = unpack(MATERIAL)
    ids = sorted(r["id"] for r in rows if r["split"] == "train")
    log = unpack(FIT / "updates.json.gz")
    for epoch in range(EPOCHS):
        assert sorted(k for b in log if b["epoch"] == epoch for k in b["rows"]) == ids
    pilot = [r for r in rows if r["split"] == "pilot"]
    assert set(fit["after"]) == {r["id"] for r in pilot}
    chosen = [
        min(
            (r for r in pilot if r["group"] == g),
            key=lambda r: fingerprint([33107, r["id"]]),
        )
        for g in sorted({r["group"] for r in pilot})
    ]
    start = monotonic()
    mx.reset_peak_memory()
    checks = []
    for r in chosen:
        p, y = r["p"], r["y"]
        cache = model.make_cache()
        for i in range(0, len(p) - 1, 128):
            model.populate_cache(mx.array([p[i : min(i + 128, len(p) - 1)]]), cache)
        sums = []
        counts = Counter()
        negative = {}
        for i, t in r["negatives"]:
            negative.setdefault(i, []).append(t)
        penalties = []
        inp = p[-1:] + y[:-1]
        for j in range(0, len(y), 8):
            logp = model(mx.array([inp[j : j + 8]]), cache)[0]
            true = mx.take_along_axis(logp, mx.array(y[j : j + 8])[:, None], axis=-1)[
                :, 0
            ]
            sums.extend((-true).tolist())
            assert mx.max(mx.abs(mx.exp(logp).sum(axis=-1) - 1)).item() < 1e-4
            for a in range(j, min(j + 8, len(y))):
                for token in negative.get(a, []):
                    margin = true[a - j] - logp[a - j, token]
                    penalties.append(mx.logaddexp(mx.array(0.0), 1 - margin).item())
                    counts[a] += 1
        ce = mean(sums)
        contrast = mean(penalties) if penalties else 0.0
        assert sum(counts.values()) == len(r["negatives"]) and len(sums) == len(y)
        expected = fit["after"][r["id"]]
        checks.append(
            dict(
                id=r["id"],
                group=r["group"],
                positions=len(sums),
                loss=ce,
                contrast=contrast,
                loss_error=abs(ce - expected[0]),
                contrast_error=abs(contrast - expected[1]),
            )
        )
    seconds = monotonic() - start
    ok = all(w["loss_error"] < 0.10 and w["contrast_error"] < 0.15 for w in checks)
    persist(
        OUT,
        dict(
            passed=ok,
            model=fit["weights"],
            fit=digest(FIT / "receipt.json"),
            updates=len(log),
            examples_per_epoch=len(ids),
            verification=checks,
            seconds=seconds,
            peak_bytes=mx.get_peak_memory(),
        ),
    )


if __name__ == "__main__":
    run()
