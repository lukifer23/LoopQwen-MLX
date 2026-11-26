"""Expose user-owned JSON data as read-only lookup callbacks for the command line."""
from .tool_session import Tool


def load_catalog(definitions):
    if not isinstance(definitions,list) or not definitions:
        raise ValueError("Supply a nonempty list of catalog definitions")
    tools=[]
    for definition in definitions:
        name=definition["name"]
        key=definition["argument"]
        rows=definition["records"]
        if not isinstance(key,str) or not key or not isinstance(rows,dict):
            raise ValueError("A catalog requires an argument name and a records object")
        def lookup(_rows=rows,_key=key,**arguments):
            return {"value":_rows[arguments[_key]]}
        parameters=dict(type="object",properties={key:{"type":"string"}},required=[key],additionalProperties=False)
        tools.append(Tool(name,definition["description"],parameters,lookup))
    return tools
