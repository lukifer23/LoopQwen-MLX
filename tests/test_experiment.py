from loopqwen.evaluation.comparison import verify_pair


def test_finished_training_controls_match():
    dense, loop = verify_pair("pilot")
    assert dense["order_hash"] == loop["order_hash"]
    assert dense["step"] == loop["step"] == 200
