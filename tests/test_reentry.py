import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
from test_future_refinement import tiny_text_decoder

from loopqwen.kernels.gdn_backend import select
from loopqwen.looped_model import LoopedModel
from loopqwen.recurrence import Span
from loopqwen.reentry import ReentryModel
from loopqwen.research.table_policy_learning import advantages, token_objective
from loopqwen.worker import acquire


def test_full_hybrid_reentry_identity_cached_states_and_policy_gradient():
    acquire()
    mx.random.seed(5143)
    net = ReentryModel(tiny_text_decoder(), 8)
    ordinary = LoopedModel(net.text, Span(0, 4, 1))
    net.eval()
    ordinary.eval()
    x = mx.array([[2, 8, 12, 19, 3, 11, 7]])
    assert mx.array_equal(ordinary(x), net(x)).item()
    assert len(tree_flatten(net.trainable_parameters())) == 9
    net.mixing = mx.full((64,), 0.4)
    cache = net.make_cache()
    a = net(x[:, :3], cache)
    b = net(x[:, 3:], cache)
    assert mx.allclose(
        mx.concatenate((a, b), axis=1), net(x), atol=4e-5, rtol=1e-4
    ).item()
    assert len(cache.slots) == 8 and all(
        cache.slots[i, 0] is not cache.slots[i, 1] for i in range(4)
    )
    assert not mx.array_equal(ordinary(x), net(x)).item()
    frozen = net.first_visit(x)
    select(net, "native")
    net.train()
    ix = mx.array([2, 3, 4])
    y = mx.array([12, 19, 3])
    (loss, nll), g = nn.value_and_grad(net, token_objective)(
        net, frozen, ix, y, 0.8, 128
    )
    assert mx.allclose(loss, nll.sum() / 128).item()
    assert all(mx.isfinite(v).all().item() for _, v in tree_flatten(g))
    assert all(mx.abs(t["up"]["weight"]).sum().item() > 0 for t in g["residuals"])
    assert (
        abs(sum(advantages([0.0, 0.2, 0.7, 1.0]))) < 1e-6
        and advantages([1.0] * 4) == [0.0] * 4
    )


def test_sampled_native_tokens_match_recorded_log_probabilities_and_human_changed_result(
    tmp_path,
):
    import math

    from loopqwen.io import read
    from loopqwen.research.table_policy_learning import sample
    from loopqwen.research.table_worlds import SOURCE, prepare_tables, query_results

    acquire()
    # The real source asks for the first listed name. Missing-name markers precede it in the changed world.
    w = next(w for w in read(SOURCE)["cohorts"]["train"] if w["id"] == "nt-5572")
    prepare_tables(tmp_path, [w])
    assert [
        x["correct"]
        for x in query_results(
            'SELECT "Name" FROM records ORDER BY rowid LIMIT 1', w, tmp_path
        )
    ] == [True, False]
    assert all(
        x["correct"]
        for x in query_results(
            """SELECT "Name" FROM records WHERE "Name" <> '-' ORDER BY rowid LIMIT 1""",
            w,
            tmp_path,
        )
    )
    net = ReentryModel(tiny_text_decoder(), 8)
    net.mixing = mx.full((64,), 0.3)

    class Guard:
        def check(self):
            return None

    prefix = [1, 7, 5, 14]
    temp = 0.8
    rows = sample(net, prefix, {127}, 4, 5, temp, 73291, Guard())
    assert len({tuple(r["tokens"]) for r in rows}) > 1
    for r in rows:
        y = r["tokens"]
        seq = mx.array([prefix + y[:-1]])
        logits = net(seq)[0, len(prefix) - 1 :].astype(mx.float32) / temp
        actual = nn.losses.cross_entropy(logits, mx.array(y), reduction="none")
        assert mx.allclose(actual, -mx.array(r["behavior_logp"]), atol=2e-4).item()
        assert all(math.isfinite(v) for v in r["behavior_logp"])
