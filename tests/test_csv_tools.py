import pytest

from loopqwen.csv_tools import tools


def test_csv_quoted_rows_bom_and_file_identity(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_bytes(b'\xef\xbb\xbfname,notes\r\n"A,B","two\nlines"\r\n\r\nC,\r\n')
    count, cell = tools([path])
    assert count.callback(path.name) == {"value": 2}
    assert cell.callback(path.name, 1, "notes") == {"value": "two\nlines"}
    assert cell.callback(path.name, 2, "notes") == {"value": ""}
    with pytest.raises(ValueError, match="outside"):
        cell.callback(path.name, 3, "name")
    with pytest.raises(ValueError, match="supplied"):
        count.callback("../sample.csv")
    path.write_text("name,notes\nchanged,text\n")
    with pytest.raises(ValueError, match="changed"):
        count.callback(path.name)


def test_csv_ambiguous_columns_and_file_names(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("a,a\n1,2\n")
    with pytest.raises(ValueError, match="distinct"):
        tools([path, path])
    with pytest.raises(ValueError, match="exactly one"):
        tools([path])[1].callback(path.name, 1, "a")


def test_clarification_requires_the_missing_input_before_the_correct_callback():
    from loopqwen.research.action_preferences import grade

    call = {
        "name": "csv_cell",
        "arguments": {"row": 7, "column": "size", "document": "a.csv"},
    }
    question = {"kind": "missing", "missing": "row", "call": call, "gold": "32"}
    first = {"status": "answered", "answer": "Which row should I read?", "turns": [{}]}
    last = {
        "status": "answered",
        "answer": "32",
        "turns": [{"call": call, "observation": {"value": "32"}}],
    }
    assert grade(question, [first, last])["success"]
    assert not grade(question, [{**first, "answer": "Which file?"}, last])["success"]
    assert not grade(question, [{**first, "turns": [{"call": call}]}, last])["success"]
    wrong = grade(question, [first, {**last, "answer": "32 or 42"}])
    assert wrong["answer"] and not wrong["success"]
