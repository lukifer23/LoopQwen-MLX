def test_verified_trace_requires_all_transitions():
    from loopqwen.training.verified_trace import verdict

    mapping = [5, 4, 7, 6, 0, 2, 1, 3]
    assert verdict(mapping, 0, "5, 2, 7", "line")["verified"]
    wrong = verdict(mapping, 0, "4, 2, 7", "eos")
    assert wrong["final_correct"] and wrong["status"] == "incorrect_transition"
    assert not verdict(mapping, 0, "5, 2, 7", "cap")["verified"]
    assert verdict(mapping, 0, "Final answer: 7", "eos")["status"] == "missing_steps"


def test_curricula_preserve_strata_and_ignore_candidate_order():
    from loopqwen.data.curriculum import select
    from pathlib import Path
    import gzip, json

    p = Path(__file__).resolve().parents[1] / "docs/measurements/rsi1"
    if not p.exists():
        import pytest

        pytest.skip("Optional measured RSI archive absent")

    def load(f):
        return json.loads(gzip.decompress((p / f).read_bytes()))

    rows = load("pools.json.gz")["pool1"]
    diag = {}
    for r in load("adaptive1-diagnosis.json.gz")["records"]:
        diag.setdefault(r["map_id"], []).append(r)
    for name, observations in [("static", None), ("adaptive", diag)]:
        chosen = select(list(reversed(rows)), observations, 1)
        expected = json.loads((p / (name + "1-curriculum.json")).read_bytes())[
            "selected"
        ]
        assert [(r["map_id"], why) for r, why in chosen] == [
            (r["id"], r["reason"]) for r in expected
        ]
