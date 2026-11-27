"""Reserve one change to the failed residual mechanism and its still unopened primary."""

from pathlib import Path

from loopqwen.io import digest, read
from scripts.register_content_read import (
    EX,
    FIT as PREVIOUS,
    PRIMARY as CLOSED,
    PANEL,
    PLAN as ORIGINAL,
    RESULT,
)
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

FIT = "bounded-read-fit-01"
PRIMARY = "bounded-read-transfer-01"
PLAN = Path("research/diagnostics/bounded-read-direct.json")


def run():
    previous = RESULT / PREVIOUS
    for p in (previous / "checksums.json", previous / "decision.json", ORIGINAL):
        committed(p)
    assert not read(previous / "decision.json")["passed"]
    assert not (Path("outputs/research") / CLOSED / "candidate.json.gz").exists()
    direct = read(ORIGINAL)
    direct["question"] = (
        "Can a per-position residual bound preserve the two-read parent's reasoning while its learned source query improves grounded answers on this unopened panel?"
    )
    direct["unopened_predecessor"] = digest(previous / "decision.json")
    persist(PLAN, direct)
    s = read(previous / "registration.json")
    s.update(
        id=FIT,
        bounded_write=True,
        feature_source=PREVIOUS,
        question="Does limiting each source write to 1/16 of the receiving state's RMS prevent early distribution collapse and preserve usable learned content?",
        protocol=s["protocol"]
        + " Start afresh from the same preceding learner-prefix checkpoint and seed, using only the identical frozen backbone feature files from this unsuccessful intervention. Reuse no failed learned tensors or optimizer state. Apply RMS normalization to the recalled vector and a tanh update bounded coordinatewise in FP32 by 1/16 of the current token state's RMS. Keep model, probe set, full vocabulary, optimizer, source order, 921 updates and all eleven criteria fixed. Additionally require every batch KL during the first sixteen updates below 0.5 nats/token. Stop the fit immediately if that divergence limit is reached. The upper native time budget includes hashing all 1800 feature files; report the original extraction cost separately.",
    )
    s["sources"].update(
        {
            str(previous / "checksums.json"): digest(previous / "checksums.json"),
            str(PLAN): digest(PLAN),
        }
    )
    persist(EX / (FIT + ".json"), s)
    t = read(EX / (CLOSED + ".json"))
    t.update(id=PRIMARY, fit_stage=FIT, bounded_write=True, question=direct["question"])
    t["sources"].update(
        {
            str(EX / (FIT + ".json")): digest(EX / (FIT + ".json")),
            str(PLAN): digest(PLAN),
        }
    )
    persist(EX / (PRIMARY + ".json"), t)


if __name__ == "__main__":
    run()
