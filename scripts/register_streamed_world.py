"""Register a memory-bounded continuation on the same immutable exposure and untouched evaluation."""

from pathlib import Path

from loopqwen.io import digest, read
from scripts.table_policy_material import persist

EX = Path("research/experiments")
OLD = Path("research/results/contrastive-world-fit-01")
FIT = "streamed-world-fit-01"
PRIMARY = "streamed-world-transfer-01"


def register():
    assert read(OLD / "decision.json") == {
        "passed": False,
        "status": "interrupted",
        "reason": "System memory pressure level 2, 3.62 GiB available (floor 1.5 GiB)",
        "durable_updates": 1152,
    }
    source = read(OLD / "registration.json")
    assert not (
        OLD.parent / "contrastive-world-transfer-01" / "candidate.json.gz"
    ).exists()
    additions = {
        str(p): digest(p)
        for p in [
            OLD / "checksums.json",
            OLD / "partial-optimizer.safetensors",
            OLD / "partial-adapters.safetensors",
            OLD / "partial-state.json",
            OLD.parent / "shared-query-features-01" / "checksums.json",
            Path("docs/measurements/shared-query-partition.json"),
        ]
    }
    s = {
        **source,
        "id": FIT,
        "resume_archive": str(OLD),
        "resume_step": 1152,
        "position_tile": 48,
        "gradient_cosine": 0.995,
        "gradient_tolerance": 0.04,
        "objective_tolerance": 0.005,
        "sources": {**source["sources"], **additions},
        "limits": {"wall_seconds": 3500, "peak_gib": 2.25, "available_gib": 1.5},
        "question": "Can an exact chain-rule decomposition along the position axis finish fitting with a smaller peak on an 18 GiB Mac, preserving the same optimizer, examples, full vocabulary and training gates?",
        "protocol": source["protocol"]
        + " This is a new resource study. The preceding run closed on the system warning at 1152 durable updates. Restore its hash-verified weights, AdamW moments and entire ordered update log. On three fixed training microbatches require the original full-vocabulary objective and each tiled version to agree in all four loss sums to 0.005 relative (denominator max(1,abs(reference))), gradient cosine at least 0.995 and relative gradient difference at most 0.04. Then complete only remaining updates, to absolute update 3392, and the same full unassisted generation gates. Every positive label and paired difference is retained, with the exact preference derivative accumulated on all affected positions. Stop on memory pressure, limit or missed numerical/quality criteria. Whole-experiment training time including the preceding 790.59 seconds must stay under its original 4380 seconds. No new examples, retuning or held predictions enter this fit.",
    }
    persist(EX / (FIT + ".json"), s)
    old = read(EX / "contrastive-world-transfer-01.json")
    p = {
        **old,
        "id": PRIMARY,
        "feature_stage": FIT,
        "sources": {
            **old["sources"],
            str(EX / (FIT + ".json")): digest(EX / (FIT + ".json")),
            str(OLD / "checksums.json"): digest(OLD / "checksums.json"),
        },
        "protocol": old["protocol"]
        + " This new pair requires the streamed continuation to qualify and its fitted checkpoint to reload independently. The original held run never started. Keep its 148 prompts, order, actual clocks, unobstructed reasoning, metric thresholds, blind service and qualitative readings, complete 128K-plus histories, maximum budget and all stop rules. No result of the closed attempts counts towards qualification.",
    }
    persist(EX / (PRIMARY + ".json"), p)


if __name__ == "__main__":
    register()
