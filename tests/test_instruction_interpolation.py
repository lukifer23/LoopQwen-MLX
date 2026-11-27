from loopqwen.research.instruction_interpolation import referent, score


def test_overlapping_referents():
    choices = ["Mary", "Mary’s daughter"]
    assert referent("Mary’s daughter.", choices) == "mary's daughter"
    assert referent("The answer is Mary.", choices) == "mary"
    assert referent("Mary and Mary's daughter", choices) is None


def test_unobserved_file_value_cannot_pass():
    row = {
        "family": "inventory",
        "keys": ["pkg-1.0.whl", "pkg-2.0.whl"],
        "values": [43, 89],
        "gold": ["43", "46"],
        "document": "inventory.csv",
    }
    tr = {"answer": "pkg-1.0.whl is 43 bytes.", "status": "answered", "turns": []}
    assert score(row, 0, tr)["correct_answer"]
    assert not score(row, 0, tr)["success"]
    call = {
        "name": "csv_lookup",
        "arguments": {
            "document": "inventory.csv",
            "key_column": "filename",
            "key": row["keys"][0],
            "value_column": "bytes",
        },
    }
    tr["turns"] = [{"call": call, "observation": {"value": "43"}}]
    assert score(row, 0, tr)["success"]
    tr["turns"][0]["observation"] = {"value": "42"}
    assert not score(row, 0, tr)["success"]
