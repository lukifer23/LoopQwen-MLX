import json

import pytest
from jsonschema import ValidationError, validate

from loopqwen.research.tool_worlds import make, score, tools_for
from loopqwen.tool_session import decode_action


@pytest.mark.parametrize("text", [
    '{"name":"x","name":"y","arguments":{}}',
    '{"name":"x","arguments":[]}',
    '{"name":"x","arguments":{},"other":1}',
    '{"name":"x","arguments":{}} trailing text',
])
def test_reject_ambiguous_actions(text):
    with pytest.raises(ValueError):
        decode_action(text)


def test_real_world_link_and_value_require_the_observation():
    row = next(w for w in make(3421,["garden"],1,"train","exercise") if w["kind"]=="chain")
    tools={t.name:t for t in tools_for(row)}
    link,lookup=row["expected"]
    first=tools[link["name"]].callback(**link["arguments"])
    assert first["value"]==row["places"][0]
    tool=tools[lookup["name"]]
    assert str(tool.callback(**{row["args"][1]:first["value"]})["value"])==row["gold"]
    with pytest.raises(ValidationError):
        validate({"not_a_parameter":first["value"]},tool.parameters)
    assert not score(row,{"answer":row["gold"],"status":"answered","turns":[]})["success"]


def test_parse_object_and_concise_answer():
    assert decode_action("Which room?") is None
    call={"name":"area","arguments":{"width":4}}
    assert decode_action(json.dumps(call))==call
