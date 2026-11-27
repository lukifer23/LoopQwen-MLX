"""Measure whether trained long-source decisions depend on the labeled evidence during the two reads."""

import argparse
import math
from pathlib import Path
from time import monotonic
from statistics import mean

import mlx.core as mx
from loopqwen.io import read, digest
from loopqwen.shared_query_assistant import SharedQueryAssistant
from loopqwen.worker import resources
from scripts.changed_locale_fit import PLAN as FIT, OUT as STUDY
from scripts.full_document_features import DIR
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

PLAN = Path("research/diagnostics/locale-evidence-intervention.json")
OUT = Path("docs/measurements/changed-locale-fit/evidence-intervention.json")


def run(freeze):
    proof = STUDY / "verification.json"
    policy = STUDY / "research-policy"
    for p in (proof, FIT, policy / "checksums.json", STUDY / "receipt.json"):
        committed(p)
    spec = dict(
        training=read(FIT)["train"],
        measure=digest(STUDY / "receipt.json"),
        weights=read(proof)["weights"],
        implementation=digest(Path(__file__)),
        protocol="A training-source intervention only. Never load any held-field feature. For each of the 12 original training source/version presentations, examine the first token where the two original answers differ. Hold the full native ordinary representation, all causal positions and learned weights fixed. Mask the true annotated value span on all additional read operations at this output position; compare a matched-width mask 64 original tokens before that span in exactly the same document and an intact reference. Neither mask changes the source tokens, ordinary cache write, position, or ordinary backbone history. Compare the correct-versus-alternative logit margin and full-vocabulary target NLL. Require intact logits to reproduce the archived training margins within .15, finite complete measurements, and a larger target margin after the decoy mask than after the evidence mask on >=9/12 with mean advantage >.10 for positive source dependence. Limit 120 native seconds and 3 GiB peak. No free answer or independent semantic credit; no additional runs on prior held sources.",
    )
    if freeze:
        persist(PLAN, spec)
        return
    committed(PLAN)
    assert spec == read(PLAN)
    folder = DIR / digest(FIT)
    fit = read(STUDY / "receipt.json")
    start = monotonic()
    mx.reset_peak_memory()
    actor = SharedQueryAssistant("exports/tool-language-packed-05", policy)
    net = actor._tool_model
    rows = []
    failure = None
    for v, doc in enumerate(fit["features"]):
        shared = mx.load(str(folder / f"v{v}-prefix.safetensors"))
        for key in spec["training"]:
            row = doc["rows"][key]
            other = fit["features"][1 - v]["rows"][key]["labels"]
            j = next(i for i, (a, b) in enumerate(zip(row["labels"], other)) if a != b)
            target = row["labels"][j]
            alternative = other[j]
            z = mx.load(str(folder / f"v{v}-{key}.safetensors"))
            pos = row["offset"] + j
            n = pos + 1
            k = mx.concatenate([shared["k"], z["tail_keys"]], axis=2)[:, :, :n]
            value = mx.concatenate([shared["v"], z["tail_values"]], axis=2)[:, :, :n]
            begin, end = row["span"]
            assert 0 < begin - 64 < end - 64 < begin < end < n and n > 131072
            scores = {}
            for label, span in (
                ("intact", None),
                ("source", (begin, end)),
                ("nearby", (begin - 64, end - 64)),
            ):
                mask = None
                if span:
                    ix = mx.arange(n)
                    mask = ((ix < span[0]) | (ix >= span[1]))[None, None, None, :]
                h = net.refine(
                    z["pre"][:, j : j + 1],
                    z["ordinary"][:, j : j + 1],
                    k,
                    value,
                    pos,
                    mask,
                )
                logits = net.project(net.text.model.norm(h))[0, 0].astype(mx.float32)
                scores[label] = dict(
                    margin=(logits[target] - logits[alternative]).item(),
                    nll=(mx.logsumexp(logits) - logits[target]).item(),
                )
            reference = next(
                r["margin"]
                for r in fit["training"]["after"]
                if r["key"] == key and r["version"] == v
            )
            rows.append(
                dict(
                    version=v,
                    key=key,
                    decision=j,
                    target=target,
                    alternative=alternative,
                    source=row["span"],
                    delta=scores["nearby"]["margin"] - scores["source"]["margin"],
                    scores=scores,
                    intact_error=abs(scores["intact"]["margin"] - reference),
                )
            )
            if monotonic() - start > 120 or mx.get_peak_memory() > 3 * 2**30:
                failure = "native resource limit"
                break
        if failure:
            break
    verified = (
        not failure
        and len(rows) == 12
        and max(r["intact_error"] for r in rows) < 0.15
        and all(math.isfinite(r["delta"]) for r in rows)
    )
    persist(
        OUT,
        dict(
            passed=bool(verified),
            source_dependence=bool(
                verified
                and sum(r["delta"] > 0 for r in rows) >= 9
                and mean(r["delta"] for r in rows) > 0.1
            ),
            plan=digest(PLAN),
            rows=rows,
            stop=failure,
            seconds=monotonic() - start,
            peak_bytes=mx.get_peak_memory(),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "measure"))
    run(parser.parse_args().action == "freeze")
