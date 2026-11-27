from loopqwen.research.shared_query_transfer import score


def test_report_scoring_requires_an_actual_nonidentity_calculation_and_consistent_final():
    w = {
        "family": "report",
        "gold": "7",
        "tolerance": "0",
        "observed": {"results": ["7"]},
    }

    def result(expression, value, answer):
        t = {
            "call": {"name": "calculate", "arguments": {"steps": [expression]}},
            "observation": {"value": value, "results": [value]},
        }
        return score(w, {"status": "answered", "turns": [t], "answer": answer})

    assert result("3+4", "7", "7")["success"]
    assert not result("7+0", "7", "7")["arithmetic_path"]
    assert not result("3+3", "6", "7")["success"]
    wrong = result("3+4", "7", "12")
    assert wrong["arithmetic_path"] and not wrong["success"]


def test_full_explanations_can_state_the_conclusion_and_cannot_gain_keyword_credit():
    w = {"family": "strategy", "label": False}

    def answered(text):
        return score(w, {"status": "answered", "answer": text, "turns": []})["success"]

    assert answered("**No**, the dates do not overlap.")
    assert not answered("Yes, even though an earlier account said no.")
    assert not answered("The notes contain yes and no.")
    assert not score(w, {"status": "token_limit", "answer": "No.", "turns": []})[
        "success"
    ]


def test_successor_reads_both_reviews_of_a_completed_pair_and_accepts_an_archived_partial(
    tmp_path, monkeypatch
):
    import pytest

    from loopqwen.io import save
    from loopqwen.research import shared_query_features as features

    monkeypatch.chdir(tmp_path)
    root = tmp_path / "results"
    monkeypatch.setattr(features, "RESULTS", root)
    for stage in (
        "contrastive-table-features-01",
        "contrastive-language-features-01",
        "contrastive-world-fit-01",
        "contrastive-world-transfer-01",
    ):
        path = root / stage
        save(path / "checksums.json", {})
        save(path / "decision.json", {"passed": True})
    save(path / "decision.json", {"passed": False})
    (path / "candidate.json.gz").touch()
    assert features.predecessor()["stage"] == stage
    (path / "control.json.gz").touch()
    reports = tmp_path / "docs" / "measurements"
    service = reports / "contrastive-services-reading.json"
    qual = reports / "contrastive-qualitative-reading.json"
    save(service, {"passed": False})
    with pytest.raises(FileNotFoundError):
        features.predecessor()
    save(qual, {"source_pairs": {}})
    assert len(features.predecessor()["source_readings"]) == 2
    save(path / "decision.json", {"passed": True})
    assert features.predecessor()["stage"] == stage
    save(service, {"passed": True})
    with pytest.raises(ValueError, match="Replicate"):
        features.predecessor()


def test_staged_context_scoring_requires_every_actual_prefix_and_decode_row():
    from copy import deepcopy
    import pytest
    from loopqwen.identity import fingerprint

    ids = [7] * 131073
    item = {
        "family": "locale",
        "initial_ids": ids,
        "initial_sha256": fingerprint(ids),
        "gold": "2 tbsp",
    }
    turn = {
        "prompt_ids": ids,
        "token_ids": [5, 6],
        "cache_state_blocks": 0,
        "scratch_bytes": 2048,
        "kv_history": [
            {"layer": i, "turn": 0, "offset": len(ids) + 1, "retained": len(ids) + 1}
            for i in (3, 7, 11, 15, 19, 23)
        ],
    }
    answer = {"status": "answered", "answer": "2 tbsp", "turns": [turn]}
    assert score(item, answer)["success"]
    assert not score(item, {**answer, "answer": "2 tablespoons"})["success"]
    broken = deepcopy(answer)
    broken["turns"][0]["kv_history"][-1]["retained"] -= 1
    with pytest.raises(AssertionError):
        score(item, broken)
    ordinary = {**turn, "cache_state_blocks": (len(ids) - 1) // 128}
    ordinary.pop("scratch_bytes")
    assert score(item, {**answer, "turns": [ordinary]})["success"]
