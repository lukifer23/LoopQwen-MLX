from loopqwen.data import pointer5, pointer6
from loopqwen.evaluation.order_scores import orbit


def test_order_changes_only_presentation_and_preserves_the_registered_schedule():
    pointer6.build()
    from loopqwen.training.pointer5 import schedule

    train = pointer5.rows("train")
    for epoch, idx in [(0, 0), (0, 479), (1, 0)]:
        group = pointer6.epoch_batches(train, 109, epoch)[idx]
        n, original = schedule(109, epoch * 480 + idx, train)
        assert group == original and n == pointer6.depth(epoch * 480 + idx, epoch)
        for r in group:
            a = pointer6.ordered(r, "randomized", 109, epoch)
            assert sorted(a["order"]) == list(range(8))
            assert orbit(a["map"], a["start"], 6)[0] == r["states"]
            assert a == pointer6.ordered(r, "randomized", 109, epoch)
            assert "Hops" not in pointer5.prompt(a)
    assert orbit([1, 2, 2, 2], 0, 5) == ([1, 2, 2, 2, 2], 2, 1)


def test_map_pairing_and_independent_rescore():
    from loopqwen.evaluation.order_scores import metrics, paired
    import pytest

    r = pointer5.rows("development")[0]
    p = [[int(i == s) for i in range(8)] for s in r["states"][:3]]
    base = dict(**r, condition="canonical", probabilities=p, gold=r["states"][:3])
    altered = {**base, "condition": "random0", "order": list(reversed(range(8)))}
    result = metrics([base, altered])
    assert result["conditions"]["mean_random"]["trajectory"] == [1, 1, 1]
    assert paired(result, result)["difference"] == [0, 0, 0]
    with pytest.raises(ValueError):
        metrics([base, {**altered, "example_id": "wrong"}])


def test_pretrained_arm_initializations_match_and_extra_heads_are_independent():
    """Run on the real pinned model. Only the three-head arm has extra tensors."""
    import gc
    import hashlib

    import mlx.core as mx
    from mlx.utils import tree_flatten
    import numpy as np

    from loopqwen.training.mechanism import StudyArm, model_for

    reference = {}
    for name in pointer6.SPEC["arms"]:
        m, _, _ = model_for(StudyArm.registered(name, 109))
        params = dict(tree_flatten(m.trainable_parameters()))
        fingerprints = {
            k: hashlib.sha256(np.array(v).tobytes()).hexdigest()
            for k, v in params.items()
        }
        common = {
            k: v for k, v in fingerprints.items() if not k.startswith("extra_heads.")
        }
        if not reference:
            reference = common
        assert common == reference
        if name == "heads":
            assert (
                len(
                    {
                        fingerprints[k]
                        for k in (
                            "state_head.weight",
                            "extra_heads.0.weight",
                            "extra_heads.1.weight",
                        )
                    }
                )
                == 3
            )
        assert sum(v.size for v in params.values()) == (
            1360896 if name == "heads" else 1344512
        )
        del m, params
        gc.collect()
        mx.clear_cache()
