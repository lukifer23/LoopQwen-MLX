"""Freeze depth and structural comparisons after the recurrent multi-domain primary passes."""

from pathlib import Path
from loopqwen.io import digest, read, save


def freeze(name, spec):
    path = Path("research/experiments") / (name + ".json")
    if path.exists() and read(path) != spec:
        raise ValueError("Existing registration differs")
    if not path.exists():
        save(path, spec)


def main():
    p = Path("research/results/tool-dialogue-counterfactual-01")
    if not read(p / "decision.json")["passed"]:
        return
    q = Path("research/results/tool-dialogue-typed-depth-01")
    old = Path("research/results/tool-dialogue-typed-01")
    s = read(p / "registration.json")
    name = "tool-dialogue-counterfactual-depth-01"
    del s["retention_floors"]
    s.update(
        id=name,
        requires=str(p),
        prerequisite_sha256=digest(p / "checksums.json"),
        initial_adapter=str(q),
        initial_sha256=digest(q / "checksums.json"),
        candidate="restore:" + str(p),
        control="train_dense",
        matched_retention=True,
        net_floor=5,
        extra_bytes=24 * 2**20,
        time_ratio=1.15,
        question="Does the improved multi-domain tool system retain its gains against an equally trained single-pass sibling, and is there paired evidence for extra depth?",
        protocol="Restore the sealed two-pass tool model and train its one-pass counterpart from the earlier matched typed K1 weights. Use exactly the same 2527 training turns, seed, rank, objective, paired examples, batch size and 1000 updates. Re-execute both on the identical 132 frozen items, common chat and tool settings. Require exact replay of all candidate input/output IDs and observed tool effects. Record paired panel scores and actual time and memory. No held-out example enters either optimizer.",
        gate="For this fixed panel eligibility requires at least as many total correct answers as the equal-training K1, no panel lower by more than one, exact K2 replay, <=1.15 times K1 seconds and <=24 MiB additional peak. Separately qualify a causal reasoning gain only for >=5 net additional correct tasks with two-sided paired p<=.05 and the same safeguards. A failed eligibility gate closes replacement; a failed causal endpoint closes attribution to extra depth.",
    )
    freeze(name, s)
    name = "tool-bfcl-counterfactual-retention-01"
    freeze(
        name,
        dict(
            id=name,
            question="Does the new two-pass adapter retain established and independently sampled structural function selection against the qualified two-pass tool release?",
            source=str(p),
            archive=digest(p / "checksums.json"),
            model=str(p),
            control_model=str(old),
            control_sha256=digest(old / "checksums.json"),
            material="research/data/tool-counterfactual-bfcl.json",
            retention=True,
            reference_traces=[
                "research/results/tool-bfcl-recovery-01",
                "research/results/tool-bfcl-multiple-01",
            ],
            max_tokens=384,
            net_floor=0,
            p_max=1.0,
            time_ratio=1.20,
            protocol="Compare two frozen K2 checkpoints with the same donor and decoding protocol. Completely replay the original 24 simple and 24 multiple BFCL questions, plus 24 independently hash-selected multiple-function questions disjoint in function names from training and all earlier panels. Freeze item order and publisher typed argument scorer. Verify that every old control output matches its sealed reference. This is a retention comparison of two trained artifacts.",
            gate="Retain at least 16/24 old simple and 18/24 old competing-schema calls. On fresh schemas require at most one fewer than the prior tool model and require at least equal pooled correct across all 72. All 48 prior control traces exact, total elapsed seconds <=1.20 times previous tool weights. Any failed floor closes replacement and the joint new release. Report paired gains, losses and exact p descriptively.",
            limits=dict(wall_seconds=472, peak_gib=6.5, available_gib=1.4),
        ),
    )


if __name__ == "__main__":
    main()
