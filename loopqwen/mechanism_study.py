"""Apply the registered development gates before starting dependent stages."""

from dataclasses import replace
from .io import read, save
from .study_gates import decide_stage
from .data.pointer6 import SPEC
from .evaluation.order_scores import metrics, paired
from .evaluation.mechanism import evaluate_arm, unpack
from .evaluation.mechanism_cost import latency, late
from .evaluation.order_diagnostics import diagnose
from .training.mechanism import StudyArm, train


def report_scores(arm):
    return metrics(
        unpack(arm.output / f"orders-{arm.milestones[-1]}.json.gz")["records"]
    )


INSTALLATION = ("canonical", "shuffled", "trajectory")
PROMOTION = (*INSTALLATION, "dense", "relabel", "patch")


def installation_checks(r):
    return dict(
        canonical=min(r["canonical"]["accuracy"]) >= SPEC["minimum_canonical"],
        shuffled=min(r["mean_random"]["accuracy"]) >= SPEC["minimum_random"],
        trajectory=r["mean_random"]["trajectory"][-1]
        >= SPEC["minimum_random_trajectory"],
    )


def installed_checkpoint(arm):
    path = arm.output / f"orders-{arm.milestones[-1]}.json.gz"
    if not path.exists():
        return decide_stage(INSTALLATION, {})
    from .io import digest
    from .training.mechanism import generation

    measured = unpack(path)
    if measured["identity"]["weights"] != digest(
        generation(arm, arm.milestones[-1]) / "adapters.safetensors"
    ):
        raise ValueError("Installation evidence and current checkpoint differ")
    return decide_stage(
        INSTALLATION, installation_checks(metrics(measured["records"])["conditions"])
    )


def decision(seed, *, follow=False):
    arms = [StudyArm.registered(k, seed) for k in SPEC["arms"]]
    scores = {a.name: report_scores(a) for a in arms}
    r = scores["randomized"]["conditions"]
    checks = installation_checks(r)
    installed = decide_stage(INSTALLATION, checks)["status"] == "eligible"
    strongest = max(
        ("middle", "full", "heads"),
        key=lambda k: scores[k]["conditions"]["mean_random"]["accuracy"][-1],
    )
    checks["dense"] = (
        r["mean_random"]["accuracy"][-1]
        > scores[strongest]["conditions"]["mean_random"]["accuracy"][-1]
    )
    record = dict(
        installed=installed,
        checks=checks,
        strongest_dense=strongest,
        random_minus_canonical=paired(scores["canonical"], scores["randomized"]),
        random_minus_dense=paired(scores[strongest], scores["randomized"]),
    )
    arm = StudyArm.registered("randomized", seed)
    if follow:
        for a in arms:
            latency(a, 480)
        late(arm, 480)
    if installed and (follow or (arm.output / "causal-480.json.gz").exists()):
        causal = diagnose(arm, 480)
        checks["relabel"] = all(
            m["prediction_agreement"] / m["n"] >= SPEC["minimum_relabel"]
            for m in causal["behavior"]["relabel"]["metrics"]
        )
        effects = {
            k: (
                v["summary"]["cross/all"]["donor_transition"]
                - v["summary"]["cross/all"]["natural_donor_transition"]
            )
            / v["summary"]["cross/all"]["n"]
            for k, v in causal["interventions"].items()
        }
        checks["patch"] = min(effects.values()) >= SPEC["minimum_patch_delta"]
        record["patch_deltas"] = effects
    installation = decide_stage(INSTALLATION, checks)
    promotion = decide_stage(PROMOTION, checks)
    record["stages"] = {
        "matched_primary_evaluation": {"status": "completed"},
        "transplants_and_annealing": installation,
        "replicate_compute_match_functional_graphs_and_lm_transfer": promotion,
    }
    record["promoted"] = promotion["status"] == "eligible"
    save(arm.output.parent / "decision.json", record)
    if installed and follow:
        outcome = replace(arm, name="outcome", objective="outcome", milestones=(80,))
        train(outcome, parent=arm)
        evaluate_arm(outcome)
        latency(outcome, 80)
        diagnose(outcome, 80)
    return record
