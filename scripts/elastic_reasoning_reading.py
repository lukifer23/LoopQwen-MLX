"""Document a descriptive source reading of every enrolled qualitative generation from the closed fit."""

from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest
from scripts.assistance_deliberation import write_csv
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

ROOT = Path("research/results/elastic-world-fit-01")
SOURCE = Path("research/sources/contrastive-worlds/training.json.gz")
# Each pair has two judgments (direction delivered, unsupported addition, reason).
READINGS = {
    "abb1e12d": [
        (True, False, "Increasing exposure gathers more light."),
        (False, False, "The answer repeats increases when the exposure is shortened."),
    ],
    "0ae26ed5": [
        (True, False, "The model selects decrease when the pusher loses energy."),
        (False, False, "It again selects decrease when the pusher gains energy."),
    ],
    "746699f0": [
        (
            True,
            False,
            "The sentence explicitly relates increasing concentration to increasing collision chance.",
        ),
        (
            True,
            True,
            "It correctly states low and explains spacing. Its claimed inverse proportionality between collision probability and distance is more specific than the source supports.",
        ),
    ],
    "04b5e662": [
        (True, False, "It selects increases for adding particles."),
        (False, False, "It selects increases for deflating the balloon."),
    ],
    "79645bcc": [
        (True, False, "It says higher-frequency waves have more energy."),
        (
            True,
            False,
            "It explicitly says lower-frequency waves have less energy. Repeating this does not reverse the conclusion.",
        ),
    ],
    "9abc3f61": [
        (
            False,
            False,
            "It attributes the heavier equal volume of rocks to lower density.",
        ),
        (True, False, "The less dense choice fits the lighter feathers."),
    ],
    "abc0c00e": [
        (
            True,
            False,
            "It explicitly selects less for the greater separation, following the supplied relation.",
        ),
        (False, False, "It selects less again as the separation decreases."),
    ],
    "4332125a": [
        (
            True,
            False,
            "It explicitly recommends more series resistors to increase resistance.",
        ),
        (
            False,
            False,
            "It claims adding more series resistors lowers resistance and contradicts its preceding sentence.",
        ),
    ],
    "0b63d6e0": [
        (
            True,
            False,
            "Increases is an inflected form of the correct choice for greater depth.",
        ),
        (False, False, "It says increasing as the person returns upwards."),
    ],
    "6195f762": [
        (False, False, "No final response was delivered."),
        (True, False, "Its final sentence says a soft push produces a small distance."),
    ],
    "b8b6d2f4": [
        (
            False,
            False,
            "The headline names Evan while the explanation concludes Tony contributes more. The response contains two incompatible choices.",
        ),
        (
            True,
            False,
            "It identifies Evan and explains that the scooter contributes less under the supplied premise.",
        ),
    ],
    "66e1142b": [
        (
            True,
            True,
            "It selects more curvature, but substitutes amount of light transmitted for degree of refraction and invents a longer-path explanation.",
        ),
        (True, False, "It selects less curvature for less refraction."),
    ],
}


def run():
    committed(ROOT / "checksums.json")
    path = ROOT / "training-free-generation.json.gz"
    rows = {w["id"]: w for w in unpack(SOURCE) if w["group"] == "qualitative"}
    records = []
    for r in unpack(path):
        if r["group"] != "qualitative":
            continue
        w = rows[r["id"]]
        correct, extra, reason = READINGS[r["pair"][-8:]][w["member"]]
        records.append(
            {
                "id": r["id"],
                "pair": r["pair"],
                "request": w["messages"][0]["content"],
                "reference": r["score"]["expected"],
                "reply": r["visible"],
                "registered": r["score"]["correct"],
                "direction_delivered": correct,
                "unsupported_extra": extra,
                "clean_direction": correct and not extra,
                "reason": reason,
            }
        )
    assert len(records) == 24
    totals = {}
    for key in ("registered", "direction_delivered", "clean_direction"):
        totals[key] = {
            "individual": sum(r[key] for r in records),
            "whole_pairs": sum(
                all(r[key] for r in records if r["pair"] == k)
                for k in {r["pair"] for r in records}
            ),
        }
    write_csv(Path("paper/elastic-world-qualitative-training-reading.csv"), records)
    out = {
        "archive": digest(ROOT / "checksums.json"),
        "raw": digest(path),
        "source": digest(SOURCE),
        "totals": totals,
        "method": "Single unblinded, retrospective project-assistant judgment on all 24 TRAINING responses, assessed against their stated passages. Direction allows inflection and prose; contradictory conclusions and an empty final do not count. Clean direction additionally excludes unsupported explanatory additions. Source-relative grading does not endorse inaccurate physical premises in the dataset. No result modifies the original 0/12 registered score or opens held evaluation.",
    }
    persist(Path("docs/measurements/elastic-qualitative-training-reading.json"), out)
    print(totals)


if __name__ == "__main__":
    run()
