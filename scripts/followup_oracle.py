"""Independently recount human answer agreement and read each actual archive using two metadata readers."""

import math
import runpy
from importlib.metadata import Distribution
from pathlib import Path

from packaging.specifiers import SpecifierSet

from loopqwen.io import digest, read, save
from loopqwen.research.followup_study import (
    agreement,
    human_ceiling,
    providers,
    refs,
    score,
)
from scripts.followup_material import PANEL, TRAIN
from scripts.wheel_archive_sources import CACHE

UPSTREAM = Path("data/research_external/quac-scorer.py")
SHA = "cd9a391bbc93a6e8c62dbbd0b4945d3f915850a9dd07be80a637d9dabff30f84"
RECEIPT = Path("docs/measurements/followup-sources.json")


def audit():
    assert digest(UPSTREAM) == SHA
    scorer = runpy.run_path(str(UPSTREAM))
    checks = callbacks = wheel_tests = 0
    enrollment = read("research/sources/wheel-archives/enrollment.json")
    wheels = {w["file"]: w for cohort in enrollment["cohorts"].values() for w in cohort}
    for w in read(PANEL)["rows"]:
        if w["family"].startswith("human_"):
            labels = w["annotations"]
            target = refs(labels)
            assert target == scorer["handle_cannot"](labels)
            assert math.isclose(
                human_ceiling(target), scorer["leave_one_out"](target), abs_tol=1e-12
            )
            guesses = [
                *labels,
                w["target"],
                "An unrelated previous answer",
                w["passage"][:51],
                "",
            ]
            if w["previous_answer"]:
                guesses.append(w["previous_answer"])
            # The published abstention branch tests the special label literally.
            # A caveat or a string containing the two words still goes through its ordinary scoring path.
            for visible in (
                "Cannot answer",
                "CANNOTANSWER",
                "Cannot answer.",
                "the cannot answer",
                "cannotanswer",
            ):
                sent = "CANNOTANSWER" if visible == "Cannot answer" else visible
                wanted = scorer["leave_one_out_max"](sent, target, w["passage"])
                assert math.isclose(agreement(visible, target), wanted, abs_tol=1e-12)
            for guess in guesses:
                official = (
                    "CANNOTANSWER"
                    if guess in ("Cannot answer", "CANNOTANSWER")
                    else guess
                )
                expect = scorer["leave_one_out_max"](official, target, w["passage"])
                assert math.isclose(agreement(guess, target), expect, abs_tol=1e-12)
                checks += 1
        if w["family"] == "archive":
            (actual,) = providers(w, None)
            for key in ("current", "previous"):
                e = w[key]
                args = e["call"]["arguments"]
                path = CACHE / "wheels" / args["wheel"]
                (dist,) = Distribution.discover(path=[str(path)])
                assert digest(path) == wheels[path.name]["sha256"]
                assert [str(v) for v in dist.metadata.get_all(args["field"])] == e[
                    "observation"
                ]["values"]
                assert actual.callback(**args) == e["observation"]
                callbacks += 1
            correct = score(
                None,
                w,
                0,
                {
                    "status": "answered",
                    "answer": w["target"],
                    "turns": [w["current"], {}],
                },
            )
            assert all(
                correct[k]
                for k in (
                    "success",
                    "correct_answer",
                    "measured",
                    "verdict",
                    "constraint",
                )
            )
            assert not score(
                None,
                w,
                0,
                {
                    "status": "answered",
                    "answer": w["target"],
                    "turns": [w["previous"], {}],
                },
            )["success"]
            assert not score(
                None,
                w,
                0,
                {
                    "status": "answered",
                    "answer": w["prior_requirement"]
                    + (" | yes" if w["allowed"] else " | no"),
                    "turns": [w["current"], {}],
                },
            )["success"]
            assert (
                SpecifierSet(w["requirement"]).contains(w["interpreter"])
                == w["allowed"]
            )
            wheel_tests += 1
    receipt = {
        "training": digest(TRAIN),
        "panel": digest(PANEL),
        "source_questions": 48,
        "official_scorer_sha256": SHA,
        "official_scorer_url": "https://s3.amazonaws.com/my89public/quac/scorer.py",
        "compared_leave_one_out_predictions": checks,
        "prior_and_current_real_wheel_reads": callbacks,
        "file_bound_verdict_checks": wheel_tests,
    }
    if RECEIPT.exists():
        assert read(RECEIPT) == receipt
    else:
        save(RECEIPT, receipt)
    return receipt


if __name__ == "__main__":
    print(audit())
