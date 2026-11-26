"""Expose user-owned JSON data as read-only lookup callbacks for the command line."""

from .tool_session import Tool


def _lookup(records, key):
    def callback(**arguments):
        return {"value": records[arguments[key]]}

    return callback


def load_catalog(definitions):
    if not isinstance(definitions, list) or not definitions:
        raise ValueError("Supply a nonempty list of catalog definitions")
    tools = []
    for definition in definitions:
        name = definition["name"]
        key = definition["argument"]
        rows = definition["records"]
        if not isinstance(key, str) or not key or not isinstance(rows, dict):
            raise ValueError("A catalog requires an argument name and a records object")

        parameters = dict(
            type="object",
            properties={key: {"type": "string"}},
            required=[key],
            additionalProperties=False,
        )
        tools.append(
            Tool(name, definition["description"], parameters, _lookup(rows, key))
        )
    return tools
