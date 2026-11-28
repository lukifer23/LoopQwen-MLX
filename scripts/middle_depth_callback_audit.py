"""Independently replay every recorded short-panel tool observation and check credit."""

from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from scripts.middle_depth_budget import ROOT
from scripts.middle_depth_development import SOURCE, instance


def audit():
    cases = {r["id"]: r for r in read(SOURCE) if r.get("provider")}
    # Reference quantities from the enrolled invoices and CSV, independent of the model.
    targets = {
        "tiles": ("184", "174", "175"),
        "ink": ("190", "270", None),
        "parcels": ("Loft", "Annex", "Friday"),
    }
    output = []
    with TemporaryDirectory() as t:
        for stage in ("terminal", "quantized"):
            for row in read(ROOT / "review" / stage / "scored.json"):
                source = row["source"]
                world = row["world"]
                if row["cap"] != 512 or source not in cases:
                    continue
                path = ROOT / row["file"]
                assert digest(path) == row["sha"]
                _, providers = instance(
                    cases[source],
                    world
                    if world in ("original", "changed", "irrelevant")
                    else "original",
                    Path(t),
                )
                lookup = {p.name: p for p in providers}
                target = targets[source][
                    2 if world in ("own", "shared") else int(world == "changed")
                ]
                actual = []
                submitted = []
                for turn in unpack(path)["trace"]["turns"]:
                    calls = [turn["call"]] if "call" in turn else turn.get("calls", [])
                    obs = (
                        [turn["observation"]]
                        if "observation" in turn
                        else turn.get("observations", [])
                    )
                    submitted.extend(calls)
                    if not obs:
                        continue
                    for call, witness in zip(calls, obs, strict=True):
                        try:
                            replay = lookup[call["name"]].callback(**call["arguments"])
                        except (ValueError, SyntaxError, ArithmeticError, TypeError) as ex:
                            replay = {"error": type(ex).__name__}
                        if replay != witness:
                            raise ValueError(f"Live provider differs from {path}")
                        actual.append(replay)
                available = any(str(x.get("value")) == target for x in actual)
                if row["grade"]["correct"] and not available:
                    raise ValueError(f"Unsupported credit on {path}")
                output.append(
                    {
                        "stage": stage,
                        "arm": row["arm"],
                        "source": source,
                        "world": world,
                        "status": row["status"],
                        "target": target,
                        "submitted": submitted,
                        "replayed": actual,
                        "target_observed": available,
                        "correct": row["grade"]["correct"],
                        "file": row["file"],
                        "sha": row["sha"],
                    }
                )
    save(
        ROOT / "quantization" / "callback-audit.json",
        {
            "rows": output,
            "presentations": len(output),
            "observations": sum(len(r["replayed"]) for r in output),
            "all_provider_results_reproduced": True,
            "graded_success_without_target_observation": 0,
        },
    )
    for stage, arm in sorted({(r["stage"], r["arm"]) for r in output}):
        rows = [
            r
            for r in output
            if (r["stage"], r["arm"]) == (stage, arm)
            and r["world"] not in ("own", "shared")
        ]
        print(
            stage,
            arm,
            len(rows),
            sum(r["target_observed"] for r in rows),
            sum(r["correct"] for r in rows),
            dict(Counter(r["status"] for r in rows)),
        )


if __name__ == "__main__":
    audit()
