import pytest

from loopqwen.csv_tools import tools


def test_keyed_record(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_bytes('id,descr\r\n100,"café,\r\nblack"\r\n101,\r\n'.encode())
    assert [t.name for t in tools([path])] == ["csv_row_count", "csv_cell"]
    (lookup,) = tools([path], keyed=True)
    assert lookup.callback(path.name, "id", "100", "descr") == {
        "value": "café,\r\nblack"
    }
    assert lookup.callback(path.name, "id", "101", "descr") == {"value": ""}
    with pytest.raises(ValueError, match="No CSV record"):
        lookup.callback(path.name, "id", "0100", "descr")
    with pytest.raises(ValueError, match="supplied"):
        lookup.callback("other.csv", "id", "100", "descr")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(path.read_bytes())
    replacement.replace(path)
    with pytest.raises(ValueError, match="changed"):
        lookup.callback(path.name, "id", "100", "descr")


@pytest.mark.parametrize(
    ("source", "error"),
    [
        ("id,x\na,\na,2\n", "Multiple"),
        ("id,id,x\na,a,2\n", "unambiguous"),
        ("id,x\na,2\nb,3,4\n", "different widths"),
    ],
)
def test_keys_require_unambiguous_well_formed_records(tmp_path, source, error):
    path = tmp_path / "data.csv"
    path.write_text(source)
    (lookup,) = tools([path], keyed=True)
    with pytest.raises(ValueError, match=error):
        lookup.callback(path.name, "id", "a", "x")
