"""Cross-order, balanced-donor interventions on representations from the actual shared block."""

from pathlib import Path
from ..data.pointer6 import ordered
from ..io import digest
from .mechanism import load, packed, unpack
from .pointer5_donors import balanced
from .pointer5_diagnostics import transplant, behavioral
from ..identity import fingerprint


def diagnose(arm, step):
    m, tok, _, weight = load(arm, step)
    original = balanced(109)
    cases_digest = fingerprint(original)
    path = arm.output / f"causal-{step}.json.gz"
    ident = dict(
        weights=weight,
        cases=cases_digest,
        source=digest(Path(__file__)),
        base=digest(Path("loopqwen/evaluation/pointer5_diagnostics.py")),
        orders=digest(Path("loopqwen/data/pointer6.py")),
    )
    if path.exists():
        old = unpack(path)
        if old["identity"] != ident:
            raise ValueError("Incompatible causal audit")
        return old
    tests = {}
    for donor, recipient in (
        ("canonical", "canonical"),
        ("canonical", "random0"),
        ("random1", "canonical"),
        ("random1", "random0"),
    ):
        cases = [
            {
                **c,
                "original": ordered(c["original"], recipient, 61091),
                **{
                    k: ordered(c[k], donor, 61091)
                    for k in ("cross", "other_start", "same_state")
                },
            }
            for c in original
        ]
        trial = transplant(m, tok, "recurrent", cases, 109)
        for key, res in trial["summary"].items():
            kind, subset = key.split("/")
            items = [
                r
                for r in trial["records"]
                if r["kind"] == kind
                and (subset == "all" or r["donor_read"] == r["donor"]["states"][0])
            ]
            res["donor_map_persistence"] = sum(
                r["patch"] == r["donor_map_next"] for r in items
            )
        tests[f"{donor}/{recipient}"] = trial
    result = dict(
        identity=ident,
        interventions=tests,
        behavior=behavioral(m, tok, "recurrent", original),
    )
    packed(path, result)
    return result
