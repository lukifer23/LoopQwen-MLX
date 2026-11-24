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
