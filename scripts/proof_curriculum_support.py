"""Enumerate inclusion-minimal premise sets supporting each decisive publisher verdict."""

import hashlib
import itertools
import json
from pathlib import Path

import z3
from reasoned_evidence_oracle import Language


def run():
    source = Path("docs/measurements/reasoned-evidence-source-oracle.json")
    original = json.loads(source.read_bytes())
    rows = []
    for r in original["rows"]:
        if r["formal"] == "uncertain":
            continue
        lang = Language()
        facts = [lang.parse(v) for v in r["premises_fol"]]
        claim = lang.parse(r["conclusion_fol"])
        # Since the full premise set is satisfiable, each of its subsets is satisfiable.
        challenge = z3.Not(claim) if r["formal"] == "true" else claim
        minimal = []
        checked = 0
        for n in range(len(facts) + 1):
            for combo in itertools.combinations(range(1, len(facts) + 1), n):
                if any(set(prior).issubset(combo) for prior in minimal):
                    continue
                solver = z3.Solver()
                solver.set(timeout=5000)
                solver.add(challenge, *[facts[i - 1] for i in combo])
                result = solver.check()
                if result == z3.unknown:
                    raise ValueError((r["id"], combo, solver.reason_unknown()))
                checked += 1
                if result == z3.unsat:
                    minimal.append(combo)
        assert minimal
        rows.append(
            {
                "id": r["id"].removesuffix("-inline"),
                "verdict": r["formal"],
                "premises": len(facts),
                "minimal": [list(v) for v in minimal],
                "calls": checked,
            }
        )
    out = {
        "version": z3.get_version_string(),
        "source": hashlib.sha256(source.read_bytes()).hexdigest(),
        "claims": len(rows),
        "method": "All inclusion-minimal supports among numbered premises, by satisfiability of the opposite. Enumeration prunes supersets of proven supports. Uncertain conclusions have no decisive support.",
        "rows": rows,
    }
    path = Path("docs/measurements/proof-curriculum-minimal-support.json")
    if path.exists():
        assert json.loads(path.read_text()) == out
    else:
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(
        len(rows),
        "decisive claims;",
        sum(len(r["minimal"]) for r in rows),
        "minimal supports;",
        sum(r["calls"] for r in rows),
        "solver queries",
    )


if __name__ == "__main__":
    run()
