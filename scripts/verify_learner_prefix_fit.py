"""Reload the fitted controller, replay its physical-depth intervention, and regenerate source states."""

from pathlib import Path
from time import monotonic

import mlx.core as mx

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.research.conditional_features import CACHE
from loopqwen.research.learner_prefix_fit import probe, verify
from loopqwen.research.shared_query_features import arrays
from loopqwen.shared_query_assistant import SharedQueryAssistant
from scripts.register_learner_prefix import FIT, RESULT
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

OUT = Path("docs/measurements") / (FIT + "-cold.json")


class Guard:
    def __init__(self):
        self.start = monotonic()

    def check(self):
        if monotonic() - self.start > 180 or mx.get_peak_memory() > 3 * 2**30:
            raise RuntimeError("Independent replay exceeded its registered allowance")


def run():
    root = RESULT / FIT
    committed(root / "checksums.json")
    dec = read(root / "decision.json")
    verify(root, dec)
    if dec.get("status") == "interrupted":
        return
    s = read(root / "registration.json")
    idx = unpack(root / "feature-index.json.gz")
    folder = CACHE / FIT / digest(root / "registration.json")
    guard = Guard()
    mx.reset_peak_memory()
    for r in idx:
        assert digest(folder / r["file"]) == r["sha256"]
    all_rows = {w["id"]: w for w in unpack(Path(s["training"]))}
    originals = read(root / "probes.json")
    maxerr = 0
    samples = []
    for policy, paths in [
        (s["parent"], [("before", 2)]),
        (FIT, [("after", 2), ("one", 1)]),
    ]:
        model = SharedQueryAssistant(s["checkpoint"], RESULT / policy)._tool_model
        for phase, n in paths:
            vals = probe(model, idx, folder, s, guard, n)
            assert vals.keys() == originals[phase].keys()
            maxerr = max(
                maxerr, max(abs(v - originals[phase][k]) for k, v in vals.items())
            )
        if policy == FIT:
            for region in (
                "strategy/learner",
                "report_action/learner",
                "repair_known_schema/clean",
                "instruction/retain",
            ):
                r = min(
                    (v for v in idx if v["group"] == region),
                    key=lambda x: fingerprint(["cold", x["id"]]),
                )
                w = all_rows[r["id"]]
                val = arrays(model, w["p"], w["y"])
                ref = mx.load(str(folder / r["file"]))
                assert val.keys() == ref.keys() and all(
                    mx.array_equal(v, ref[k]).item() for k, v in val.items()
                )
                samples.append(r["id"])
        del model
        mx.clear_cache()
    guard.check()
    assert maxerr <= 0.00002
    persist(
        OUT,
        {
            "fit": digest(root / "checksums.json"),
            "cases": len(originals["after"]),
            "paths": 3,
            "feature_hashes": len(idx),
            "regenerated": samples,
            "max_abs_nll_error": maxerr,
            "seconds": monotonic() - guard.start,
            "peak_bytes": mx.get_peak_memory(),
            "passed": True,
        },
    )


if __name__ == "__main__":
    run()
