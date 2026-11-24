import json
from pathlib import Path

import pytest


def test_epoch_two_follows_absolute_cursor():
    from loopqwen.data import pointer5, pointer6
    from loopqwen.training.pointer5 import schedule

    rows = pointer5.rows("train")
    ordered = pointer6.epoch_batches(rows, 109, 1)
    for n in (480, 623, 719, 959):
        depth, group = schedule(109, n, rows)
        assert group == ordered[n - 480]
        assert depth == pointer6.depth(n, 1)
        assert any(
            pointer6.ordered(r, "randomized", 109, 1)["order"]
            != pointer6.ordered(r, "randomized", 109, 0)["order"]
            for r in group
        )


def test_exact_parent_and_resumed_cursor(monkeypatch):
    from mlx.utils import tree_flatten
    import mlx.core as mx
    from loopqwen import checkpoint
    from loopqwen.training.exposure import (
        ARM,
        S,
        parent_lineage,
        verify_optimizer,
        resume_check,
    )
    from loopqwen.training.mechanism import model_for
    from loopqwen.training.pointer5 import adam
    from loopqwen.io import digest, read

    repo = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo)
    if not (Path("checkpoints/phase6/109/randomized/at-480.json")).exists():
        pytest.skip("Exact optimizer generation is local to training host")
    parent = parent_lineage()
    model, _, _ = model_for(ARM)
    opt = adam()
    checkpoint.restore(parent["path"], model, opt)
    verify_optimizer(model, opt, 480, S["lr"])
    path = Path(parent["path"])
    original = mx.load(str(path / "optimizer.safetensors"))
    assert all(
        mx.array_equal(a, original[k]).item() for k, a in tree_flatten(opt.state)
    )
    assert digest(path / "state.json") == S["parent"]["metadata"]
    if (ARM.root / "latest.json").exists():
        state = checkpoint.restore(ARM.root, model, opt)
        verify_optimizer(model, opt, state["step"], S["lr"])
        resume_check(state, state["identity"])
        with pytest.raises(ValueError):
            resume_check(state, "different")
        with pytest.raises(ValueError):
            resume_check({**state, "next_absolute_step": 480}, state["identity"])
        assert parent_lineage() == parent


def test_numeric_visibility_and_cohort_separation():
    from loopqwen.evaluation.language7 import interpret, cohorts

    class Digits:
        def decode(self, ids):
            return "".join(str(i) for i in ids)

    assert interpret([7, 99], 97, 98, {99}, Digits())["answer"] is None
    assert interpret([7, 98, 4, 99], 97, 98, {99}, Digits())["answer"] == "4"
    assert interpret([98, 3, 97, 4, 99], 97, 98, {99}, Digits())["answer"] is None
    a, b = cohorts()
    assert len(a) == 8 and len(b) == 24
    assert not ({r["id"] for r in a} & {r["id"] for r in b})
