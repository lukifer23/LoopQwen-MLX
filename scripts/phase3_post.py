"""Run native post-training audits serially; cache successful compatible work."""

from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loopqwen.io import digest, read, save
from scripts.phase2_supervise import supervise


def main():
    out = Path("outputs/phase3")
    if sys.argv[1:] != ["--worker"]:
        return supervise([sys.executable, __file__, "--worker"], out / "study.lock")
    subprocess.run(
        [
            sys.executable,
            "scripts/phase2_supervise.py",
            "--check",
            str(out / "study.lock"),
        ],
        check=True,
        pass_fds=(int(os.environ["LOOPQWEN_STUDY_FD"]),),
    )
    study = read(out / "main.json")
    if study["milestone"] != 100000:
        raise ValueError("Complete the main experiment before post-training audits")
    tasks = [
        (
            "projection",
            ["-m", "loopqwen.benchmark.phase3_projection"],
            "projection.json",
        ),
        ("regression", ["scripts/phase3_regression.py"], "regression.json"),
        ("coefficients", ["scripts/phase3_gate_audit.py"], "gate-controls.json"),
        ("curves", ["-m", "loopqwen.evaluation.phase3_curve"], "learning-curves.json"),
        ("audit", ["-m", "loopqwen.benchmark.phase3_audit"], "native-audit.json"),
        ("fork", ["scripts/phase3_recovery.py"], "fork-audit.json"),
        (
            "matched-batch",
            ["-m", "loopqwen.benchmark.phase3_batch", "matched_instance_order"],
            "batch-matched_instance_order-None.json",
        ),
        (
            "protocol-batch",
            ["-m", "loopqwen.benchmark.phase3_batch", "protocol_and_length"],
            "batch-protocol_and_length-None.json",
        ),
        (
            "budget-batch",
            [
                "-m",
                "loopqwen.benchmark.phase3_batch",
                "protocol_and_length",
                "--cap",
                "360",
            ],
            "batch-protocol_and_length-360.json",
        ),
        (
            "compaction",
            ["-m", "loopqwen.benchmark.phase3_compaction"],
            "compaction.json",
        ),
        (
            "likelihood",
            ["-m", "loopqwen.evaluation.phase3_likelihood"],
            "answer-likelihood.json",
        ),
        (
            "transfer",
            ["-m", "loopqwen.evaluation.phase3_transfer", "transfer"],
            "fresh-development.json",
        ),
        ("export", ["-m", "loopqwen.evaluation.phase3_export"], "export-audit.json"),
        (
            "frontier",
            ["-m", "loopqwen.evaluation.phase3_transfer", "frontier"],
            "frontier.json",
        ),
    ]
    if (out / "promotion.json").is_file():
        tasks.append(
            (
                "export-250k",
                ["-m", "loopqwen.evaluation.phase3_export", "--targets", "250000"],
                "export-audit-250k.json",
            )
        )
    tasks.append(("difficulty", ["scripts/phase3_breakdowns.py"], "difficulty.json"))
    for arm in ("learn1", "learn2"):
        tasks.append(
            (
                "lr-" + arm,
                ["-m", "loopqwen.evaluation.gate_screen", arm],
                arm + "-lr-current.json",
            )
        )
    priority = (
        "regression",
        "audit",
        "fork",
        "coefficients",
        "lr-learn1",
        "lr-learn2",
        "curves",
        "transfer",
        "export",
    )
    tasks.sort(
        key=lambda task: (
            priority.index(task[0]) if task[0] in priority else len(priority)
        )
    )
    for task, args, artifact in tasks:
        pointer = out / "post" / (task + ".json")
        file = (
            Path(args[1].replace(".", "/") + ".py")
            if args[0] == "-m"
            else Path(args[0])
        )
        identity = dict(
            main=digest(out / "main.json"), command=args, source=digest(file)
        )
        if pointer.exists():
            prev = read(pointer)
            if prev["identity"] == identity and prev["artifact"] == digest(
                out / artifact
            ):
                continue
            raise ValueError(
                f"Existing diagnostic changed: {pointer}. Use an independent reproduction checkout."
            )
        subprocess.run([sys.executable, *args], check=True)
        save(pointer, dict(identity=identity, artifact=digest(out / artifact)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
