from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen.evaluation.mechanism import unpack
from loopqwen.research.contrastive_grading import attested, selected_choice
from loopqwen.research.contrastive_transfer import database, paired
from loopqwen.research.table_worlds import FILENAME
from scripts.contrastive_transfer_panel import PANEL
from scripts.contrastive_world_material import TABLES


def test_service_support_requires_two_distinct_observed_values_from_one_source_row():
    rows = [["Oak Inn", "Bristol", "true"], ["Harbor House", "Bath", None]]
    assert attested("Oak Inn in Bristol is available.", rows, [rows[0]])
    assert not attested(
        "Oak Inn in Bristol is available.", rows, [["Oak Inn"], ["Bristol"]]
    )
    assert not attested("Oak Inn in Bath.", rows, [rows[0], rows[1]])
    assert not attested("Bristol Bristol true.", rows, [rows[0]])
    assert not attested("Oaks Inn in Bristols.", rows, [rows[0]])


def test_qualitative_preference_and_whole_source_scoring():
    choices = ["faster", "slower"]
    assert (
        selected_choice("The correct answer is slower. It takes longer.", choices)
        == "slower"
    )
    assert selected_choice("Maybe faster or slower.", choices) is None
    assert selected_choice("It is not faster.", choices) is None
    q = paired(
        {"a": [[True, False], [True, True]], "b": [[False, True], [True, False]]}
    )
    assert q == {"whole": [1, 0], "p": 1.0, "losses": 1, "individual": [3, 2]}


def test_identical_held_openings_really_mount_different_sqlite_contents():
    worlds = [r for r in unpack(PANEL)["rows"] if r["family"] == "tables"]
    w = worlds[0]
    pair = [r for r in worlds if r["cluster"] == w["cluster"]]
    assert len(pair) == 2 and pair[0]["initial_ids"] == pair[1]["initial_ids"]
    statement = next(
        r["statement"]
        for r in unpack(TABLES / "training.json.gz")
        if r["id"] == w["source"]
    )
    assert pair[0]["gold"] != pair[1]["gold"]
    with TemporaryDirectory() as tmp:
        for item in pair:
            schema, query = database(item, Path(tmp))
            assert "records" in schema.callback(database_name=FILENAME)["tables"]
            observed = query.callback(database_name=FILENAME, statement=statement)
            assert str(observed["rows"][0][0]).casefold() == item["gold"].casefold()
