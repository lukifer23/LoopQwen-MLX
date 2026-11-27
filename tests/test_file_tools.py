from io import BytesIO

import pytest

from loopqwen.file_tools import line_count, tools


def test_file_line_count_matches_binary_splitlines_across_blocks():
    for s in (
        b"",
        b"no newline",
        b"\n",
        b"\r\n",
        b"\n\r",
        b"a\x0bb\x1cc",
        b"a\r\nb\n\n\rc\rd\x00\xff",
        bytes(range(256)),
    ):
        for block in (1, 2, 5, 41):
            assert line_count(BytesIO(s), block) == len(s.splitlines())


def test_only_caller_files_and_the_original_version_are_available(tmp_path):
    source = tmp_path / "entries.txt"
    source.write_bytes(b"first\r\nsecond\n")
    size, lines = tools([source, source])
    assert size.parameters["properties"]["path"]["enum"] == [str(source)]
    assert size.callback(str(source)) == {"bytes": 14}
    assert lines.callback(str(source)) == {"lines": 2}
    with pytest.raises(FileNotFoundError):
        size.callback("other")
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        lines.callback(str(source))
