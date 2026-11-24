"""Native mechanism study stages on a single supervised worker."""

import argparse
from .data.pointer6 import SPEC, build
from .training.mechanism import StudyArm, train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        choices=[
            "data",
            "train",
            "evaluate",
            "benchmark",
            "diagnostics",
            "followup",
            "replicate",
        ],
    )
    p.add_argument("--seed", type=int, default=SPEC["seed"])
    p.add_argument(
        "--arms", nargs="+", choices=list(SPEC["arms"]), default=list(SPEC["arms"])
    )
    args = p.parse_args()
    build()
    if args.stage == "followup":
        from .mechanism_study import decision

        print(decision(args.seed, follow=True), flush=True)
        return
    if args.stage == "replicate":
        from .evaluation.mechanism_replication import replicate

        replicate()
        return
    for name in args.arms:
        arm = StudyArm.registered(name, args.seed)
        if args.stage == "train":
            train(arm)
        if args.stage == "benchmark":
            from .benchmark.mechanism import bench

            bench(arm)
        if args.stage == "diagnostics":
            from .evaluation.order_diagnostics import diagnose

            diagnose(arm, arm.milestones[-1])
        if args.stage == "evaluate":
            from .evaluation.mechanism import evaluate_arm

            evaluate_arm(arm)


if __name__ == "__main__":
    main()
