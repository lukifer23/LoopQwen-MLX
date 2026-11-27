"""Compare native document branches to independent full-vocabulary teacher-forced states."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import mlx.core as mx

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.long_training_memory import DocumentMemory
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from loopqwen.research.shared_query_features import arrays
from loopqwen.shared_query_assistant import SharedQueryAssistant
from scripts.learner_prefix_material import DATA, FIT, RESULT
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

PLAN = Path("research/diagnostics/long-training-memory.json")
OUT = Path("docs/measurements/long-training-memory.json")


def material():
    rows = unpack(DATA)
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    cases = {}
    for g in ("strategy", "report_action", "repair_known_schema"):
        for w in sorted(
            (w for w in rows if w["group"] == g),
            key=lambda w: fingerprint(["branch parity", w["id"]]),
        ):
            p, y = token_pair(tok, w)
            if 270 <= len(p) <= 1200 and len(y) >= 6:
                cases[w["id"]] = (p, y[:6])
                break
    assert len(cases) == 3
    return cases, {
        "fit": digest(RESULT / FIT / "checksums.json"),
        "implementation": digest(Path("loopqwen/long_training_memory.py")),
        "verifier": digest(Path(__file__)),
        "cases": {k: fingerprint([p, y]) for k, (p, y) in cases.items()},
        "protocol": "Training engineering only. On actual trained two-read weights use three original training prompts and six subsequent target positions each. Stage the first 128 input positions then compute each question and successive targets with the original causal layers. Compare every full-vocabulary logit and top-1 token and every final key/value against independent single-sequence native extraction. Re-evaluate the original continuation after a distinct intervening continuation on the same immutable prefix. Require all 18 greedy tokens equal, max absolute logit difference <=0.20, mean absolute target NLL difference <=0.01 and exact repeatability of branch states. Stop after 120 native seconds or 3 GiB. Full-length training-source extraction requires a pass; no model success credit.",
    }


def run(freeze):
    cases, s = material()
    if freeze:
        persist(PLAN, s)
        return
    committed(PLAN)
    assert read(PLAN) == s
    before = monotonic()
    actor = SharedQueryAssistant("exports/tool-language-packed-05", RESULT / FIT)
    net = actor._tool_model
    rows = []
    for key, (p, y) in cases.items():
        with TemporaryDirectory() as tmp:
            shared = DocumentMemory(net, p[:128], tmp)
            a = shared.branch(p, y)
            # Exercise a genuinely different logical continuation in between.
            other = next((op, oy) for k, (op, oy) in cases.items() if k != key)
            shared.branch(p[:128] + other[0][-30:], other[1])
            b = shared.branch(p, y)
            assert all(
                mx.array_equal(a[k], b[k]).item()
                for k in ("pre", "ordinary", "tail_keys", "tail_values")
            )
            full = arrays(net, p, y)
            pref = mx.load(str(a["prefix_file"]))
            local = {
                name: mx.concatenate([pref[k], a[v]], axis=2)
                for name, k, v in (
                    ("keys", "k", "tail_keys"),
                    ("values", "v", "tail_values"),
                )
            }
            errors = {
                name: mx.max(
                    mx.abs(
                        (local[name] if name in local else a[name]).astype(mx.float32)
                        - full[name].astype(mx.float32)
                    )
                ).item()
                for name in ("keys", "values", "pre", "ordinary")
            }

            def pred(d):
                state = net.refine(
                    d["pre"],
                    d["ordinary"],
                    d["keys"],
                    d["values"],
                    len(p) - 1,
                    "causal",
                )
                return net.project(net.text.model.norm(state)).astype(mx.float32)

            x, z = pred(full), pred({**a, **local})

            def target(q):
                return mx.take_along_axis(
                    q - mx.logsumexp(q, axis=-1, keepdims=True),
                    mx.array(y)[None, :, None],
                    axis=-1,
                )

            rows.append(
                {
                    "id": key,
                    "prompt": len(p),
                    "n": len(y),
                    "state_max_error": errors,
                    "logits_max": mx.max(mx.abs(x - z)).item(),
                    "top_equal": mx.sum(
                        mx.argmax(x, axis=-1) == mx.argmax(z, axis=-1)
                    ).item(),
                    "nll_mae": mx.mean(mx.abs(target(x) - target(z))).item(),
                }
            )
        if monotonic() - before > 120 or mx.get_peak_memory() > 3 * 2**30:
            raise RuntimeError("Branch verification resource limit")
    passed = (
        all(r["top_equal"] == r["n"] and r["logits_max"] <= 0.2 for r in rows)
        and sum(r["nll_mae"] for r in rows) / len(rows) <= 0.01
    )
    persist(
        OUT,
        {
            "plan": digest(PLAN),
            "passed": passed,
            "rows": rows,
            "seconds": monotonic() - before,
            "peak_bytes": mx.get_peak_memory(),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("register", "measure"))
    run(p.parse_args().action == "register")
