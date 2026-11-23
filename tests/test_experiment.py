from loopqwen.evaluation.comparison import verify_pair


def test_finished_training_controls_match():
    dense, loop = verify_pair("pilot")
    assert dense["order_hash"] == loop["order_hash"]
    assert dense["step"] == loop["step"] == 200


def test_identical_initial_adapter_tensors():
    import hashlib

    import numpy as np
    from mlx.utils import tree_flatten

    from loopqwen.checkpoint import setup
    from loopqwen.config import config

    digests = []
    for name in ("pilot_dense", "pilot_loop"):
        model, _, _ = setup(config("configs/" + name + ".yaml"))
        digest = hashlib.sha256()
        for key, array in sorted(tree_flatten(model.trainable_parameters())):
            digest.update(key.encode())
            digest.update(np.asarray(array).tobytes())
        digests.append(digest.hexdigest())
        del model
    assert digests[0] == digests[1]


def test_curriculum_endpoints_and_replay():
    from loopqwen.config import config
    from loopqwen.training.trainer import depth

    c = config("configs/gradual_loop.yaml")
    assert {depth(c, 0, i) for i in range(8)} == {1}
    assert {depth(c, c["steps"] // 3 + 1, i) for i in range(8)} == {2}
    c.update(curriculum="mixed", depth_weights=[0.2, 0.6, 0.2])
    first = [depth(c, s, 0) for s in range(100)]
    assert first == [depth(c, s, 0) for s in range(100)]
    assert set(first) == {1, 2, 3}
