"""CPU checks for the objective-relevant scoring and curriculum decisions."""

import importlib.util
from pathlib import Path

from loopqwen.research.replay import select


def test_curriculum_reacts_to_measured_prefix_without_changing_static_choice():
    first = [dict(id=str(i), prefix=p) for i, p in enumerate([6, 3, 1, 0, 3, 5])]
    second = [dict(id=str(i), prefix=p) for i, p in enumerate([3, 0, 1, 3, 0, 5])]
    assert select(first, "online", 12) == ["1", "4"]
    assert select(second, "online", 12) == ["0", "3"]
    assert select(first, "static", 12) == select(second, "static", 12)


def test_arithmetic_audit_accepts_valid_chain_and_detects_wrong_computation():
    path = Path(__file__).resolve().parents[1] / "scripts/audit_gsm_equations.py"
    spec = importlib.util.spec_from_file_location("audit_gsm_equations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.check(["20/100=0.2", "168/0.2=168*5=840"], "840") == "consistent"
    assert module.check(["4*80=32", "32-32=10"], "10") == "false equality"
    assert module.check(["190*0.5=95", "95*0.4=38"], "19") == "final mismatch"


def test_replay_rescore_binds_weights_and_paired_requests():
    import pytest
    from scripts.research_archive import read, replay_outcomes

    root = Path("research/results/functional-replay-cap84")
    cohort = read(Path("research/data/functional-replay-01.json.gz"))["splits"]["dev"]
    parent = read(root / "initial.json.gz")
    static = read(root / "round1/static/eval.json.gz")
    hits, paths, terminal = replay_outcomes(
        static, cohort, 6, static["weights"], parent["records"]
    )
    assert hits[-1] == sum(terminal) == 53
    assert paths[-1] == 18
    with pytest.raises(ValueError, match="weights"):
        replay_outcomes(static, cohort, 6, parent["weights"], parent["records"])
    with pytest.raises(ValueError, match="Unpaired"):
        replay_outcomes(
            static, cohort, 6, static["weights"], reversed(parent["records"])
        )
