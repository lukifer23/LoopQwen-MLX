import sqlite3

import pytest

from loopqwen.relational_tools import tools


def test_inspect_and_join_two_tables_in_a_supplied_file(tmp_path):
    path = tmp_path / "music.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(
            "CREATE TABLE albums(id INTEGER PRIMARY KEY, title TEXT);"
            "CREATE TABLE tracks(album INTEGER REFERENCES albums(id), title TEXT);"
            "INSERT INTO albums VALUES (1,'Record');"
            "INSERT INTO tracks VALUES (1,'Opening'),(1,'Closing');"
        )
    inspect, query = tools([path])
    assert inspect.callback(path.name)["tables"]["tracks"]["foreign_keys"] == [
        ["album", "albums", "id"]
    ]
    result = query.callback(
        path.name,
        "SELECT albums.title, COUNT(*) FROM albums JOIN tracks ON albums.id=tracks.album GROUP BY albums.title",
    )
    assert result["rows"] == [["Record", 2]] and not result["truncated"]
    assert result["referenced_tables"] == ["albums", "tracks"]
    with pytest.raises(sqlite3.DatabaseError):
        query.callback(path.name, "INSERT INTO albums VALUES (2, 'Other')")
    assert query.callback(path.name, "SELECT count(*) FROM albums")["rows"] == [[1]]
    assert not query.callback(path.name, "SELECT 1")["referenced_tables"]


def test_reports_truncation_only_if_another_row_exists(tmp_path):
    path = tmp_path / "catalog.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE items(number INTEGER)")
        db.executemany("INSERT INTO items VALUES (?)", [(i,) for i in range(51)])
    _, query = tools([path])
    first = query.callback(path.name, "SELECT number FROM items ORDER BY number")
    assert len(first["rows"]) == 50 and first["truncated"]
    exactly = query.callback(
        path.name, "SELECT number FROM items WHERE number<50 ORDER BY number"
    )
    assert exactly["rows"] == first["rows"] and not exactly["truncated"]
