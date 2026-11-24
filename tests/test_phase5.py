from copy import deepcopy

from loopqwen.data.pointer5 import build, rows, prompt, cycle
from loopqwen.data.transitions import states


def test_pointer_splits_renderings_and_gold():
    meta = build()
    assert [r["prior_correct"] for r in meta["prior"]] == [53, 62, 54]
    tr, dev, confirm = [rows(k) for k in ("train", "development", "confirmation")]
    assert len({r["map_id"] for r in tr + dev + confirm}) == 4608
    for r in tr + dev:
        assert r["states"] == states(r["map"], r["start"], 6) and cycle(r) > 3
    r = dev[0]
    changed = deepcopy(r)
    changed["states"] = [-1] * 6
    assert prompt(r) == prompt(changed)
    assert "Hops" not in prompt(r) and prompt(r, 1) != prompt(r, 3)
    for c in rows("cases"):
        assert c["original"]["states"][:3] == c["off_path"]["states"][:3]
        assert c["original"]["states"][1] != c["on_path"]["states"][1]
        assert c["original"]["map_id"] == c["other_start"]["map_id"]
        assert c["original"]["example_id"] != c["other_start"]["example_id"]


def test_real_pointer_gradients_splitting_and_resume(tmp_path):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from loopqwen import checkpoint
    from loopqwen.training.pointer5 import (
        architecture,
        request_rows,
        inputs,
        outputs,
        make_update,
        adam,
    )

    m, t, _, _ = architecture(109, "recurrent", "off")
    m.set_dtype(mx.float32)
    m.train()
    samples = rows("development")[:2]
    group = request_rows(samples, 3, "recurrent", t, {})
    x, last = inputs(group)
    y = mx.array([r["states"][:3] for r in samples])

    def objective(m, separate):
        scores = (
            [outputs(m, x, last, k, "recurrent")[-1] for k in (1, 2, 3)]
            if separate
            else outputs(m, x, last, 3, "recurrent")
        )
        return (
            sum(
                nn.losses.cross_entropy(z, y[:, j], reduction="mean")
                for j, z in enumerate(scores)
            )
            / 3
        )

    fn = nn.value_and_grad(m, objective)
    a, g = fn(m, False)
    b, h = fn(m, True)

    def vector(tree):
        return mx.concatenate([w.reshape(-1) for _, w in tree_flatten(tree)])

    assert abs(a.item() - b.item()) < 1e-5
    assert 0 < mx.linalg.norm(vector(g["text"])).item()
    assert (
        mx.linalg.norm(vector(g) - vector(h)) / mx.linalg.norm(vector(h))
    ).item() < 0.001
    opt = adam()
    init = m.trainable_parameters()
    whole = make_update(m, opt, "recurrent")(group, 3, 384)
    expected = outputs(m, x, last, 3, "recurrent")[-1]
    mx.eval(expected)
    m.update(init)
    opt = adam()
    split = make_update(m, opt, "recurrent")(group, 3, x.shape[1])
    actual = outputs(m, x, last, 3, "recurrent")[-1]
    mx.eval(actual)
    assert whole["labels"] == split["labels"] == 6 and len(split["shapes"]) == 2
    assert max(abs(a - b) for a, b in zip(whole["ce"], split["ce"])) < 1e-4
    assert (mx.linalg.norm(actual - expected) / mx.linalg.norm(expected)).item() < 0.002
    checkpoint.write(tmp_path, m, opt, dict(step=1))
    original = dict(tree_flatten(opt.state))
    mx.eval(original)
    restored = adam()
    checkpoint.restore(tmp_path, m, restored)
    actual_state = dict(tree_flatten(restored.state))
    assert original.keys() == actual_state.keys()
    assert all(mx.array_equal(v, actual_state[k]).item() for k, v in original.items())
    assert mx.array_equal(actual, outputs(m, x, last, 3, "recurrent")[-1]).item()
