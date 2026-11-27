"""Reserve the remaining original budget for a memory-pressure-aware completion of the same fit."""

from pathlib import Path

from loopqwen.io import digest, read
from scripts.table_policy_material import persist

EX = Path("research/experiments")
OLD = Path("research/results/streamed-world-fit-01")
FIT = "elastic-world-fit-01"
PRIMARY = "elastic-world-transfer-01"


def register():
    old = read(OLD / "registration.json")
    decision = read(OLD / "decision.json")
    assert decision["status"] == "interrupted" and decision["durable_updates"] == 1920
    prior = {}
    for window in Path("research/sessions").glob("window-*.json"):
        for job in read(window)["jobs"]:
            if job["job"] in ("contrastive-world-fit-01", old["id"]):
                prior[job["job"]] = job["seconds"]
    assert len(prior) == 2
    remaining = int(4380 - sum(prior.values()))
    sources = {
        str(p): digest(p)
        for p in [
            OLD / "checksums.json",
            OLD / "partial-optimizer.safetensors",
            OLD / "partial-adapters.safetensors",
            OLD / "partial-state.json",
        ]
    }
    spec = {
        **old,
        "id": FIT,
        "resume_step": 1920,
        "resume_archive": str(OLD),
        "earlier_native_seconds": prior,
        "max_pressure_stops": 6,
        "checkpoint_interval": 16,
        "limits": {**old["limits"], "wall_seconds": remaining},
        "sources": {**old["sources"], **sources},
        "question": "Can a native worker that relinquishes its allocation during external memory pressure finish the identical optimizer schedule and unassisted generation gates within the original cumulative compute budget?",
        "protocol": old["protocol"]
        + " The second preceding attempt retained 1920 updates. Continue from those exact optimizer and model bytes. Hold the same original 4380 cumulative native seconds across all attempts, 2.25 GiB peak per worker, 1.5 GiB available floor and normal system pressure. Within one two-hour session, exit the worker on every pressure warning, resume only after ten consecutive normal seconds, at most six pressure-triggered restarts. Persist each complete 16-update block and completed generation case. All actual processes and discarded work are charged. Release frozen transformer layers during optimization of saved features and reload the original full model for free generations. Keep all training tasks, thresholds and the frozen direct comparison. Terminate permanently on non-pressure failures, exhausted budget or repeated warnings.",
    }
    persist(EX / (FIT + ".json"), spec)
    base = read(EX / "streamed-world-transfer-01.json")
    persist(
        EX / (PRIMARY + ".json"),
        {
            **base,
            "id": PRIMARY,
            "feature_stage": FIT,
            "cold_required": True,
            "sources": {
                **base["sources"],
                str(EX / (FIT + ".json")): digest(EX / (FIT + ".json")),
            },
            "protocol": base["protocol"]
            + " This pair uses only the independently restored passing pressure-aware completion. Neither predecessor generated a held response.",
        },
    )


if __name__ == "__main__":
    register()
