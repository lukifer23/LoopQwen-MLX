"""Register the same complete pair after qualifying full-precision staged attention."""

from pathlib import Path

from loopqwen.io import digest, read
from scripts.register_repair_query import EX, FIT, PANEL, PLAN, PRIMARY, RESULT
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed
from scripts.verify_staged_query import PLAN as NUMERICS
from scripts.verify_staged_query import REPORT

STAGE = "staged-query-transfer-01"


def register():
    old = RESULT / PRIMARY
    for path in (
        old / "checksums.json",
        RESULT / FIT / "checksums.json",
        REPORT,
        NUMERICS,
        PLAN,
        PANEL,
    ):
        committed(path)
    source = read(old / "registration.json")
    d = read(old / "decision.json")
    assert (
        d["status"] == "interrupted"
        and not d["passed"]
        and d["durable"] == {"candidate": 5}
    )
    assert not (old / "control.json.gz").exists()
    receipt = read(REPORT)
    assert receipt["passed"] and receipt["plan"] == digest(NUMERICS)
    assert (
        receipt["long"]["prompt"] > 131072 and receipt["long"]["read_only_visits"] == 8
    )
    spec = dict(source)
    paths = [
        REPORT,
        NUMERICS,
        old / "checksums.json",
        RESULT / FIT / "checksums.json",
        Path("docs/measurements") / (FIT + "-cold.json"),
    ]
    spec.update(
        id=STAGE,
        sources={**source["sources"], **{str(p): digest(p) for p in paths}},
        staging_receipt=str(REPORT),
        stage_from_tokens=32768,
        scratch_gib=4,
        max_pressure_restarts=4,
        max_attempts_per_item=2,
        intermediate_window_archive="research/sessions/staged-query-candidate",
        seconds_per_arm=4380,
        limits={**source["limits"], "wall_seconds": 8760},
        question="Can lossless 16-bit staging complete the entire original pair and preserve user-facing accuracy, physical depth and measured performance under host memory pressure?",
        protocol=source["protocol"]
        + " Replay both complete arms from the first item. Apply grouped full-precision disk staging to both arms on openings of 32768 or more tokens. Preserve all six attention histories, folded final recurrent reads, official thinking prompts, pinned weights, scoring, identical order and the original blinded reading. Require positive scratch use below 4 GiB on each staged turn. The original incomplete study used 102.277 seconds. Reserve at most 4380 native seconds per model in two independently closed windows, 8760 seconds total. Both models pay all staging I/O. The original incomplete attempt and separate engineering qualification are reported in addition. Record failed request attempts separately and add all their elapsed time to the recovered item's wall and first-response time, with the maximum peak retained. Never repeat a finished item. On system pressure release the process and require ten nominal seconds. Stop on the second failed attempt for the same item, after four pressure restarts, two hours of elapsed availability per arm, a resource cap or any failed original gate. Read no individual scores or aggregate product outcomes before the full committed masked review.",
    )
    persist(EX / (STAGE + ".json"), spec)


if __name__ == "__main__":
    register()
