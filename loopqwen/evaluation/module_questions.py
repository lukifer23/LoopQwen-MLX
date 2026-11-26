"""Read top-level Python standard-library source files installed in this interpreter."""

import sysconfig
from pathlib import Path

from ..io import digest
from ..tool_session import Tool


def sources():
    return {
        p.stem: p
        for p in Path(sysconfig.get_path("stdlib")).glob("*.py")
        if p.is_file() and not p.stem.startswith("_")
    }


def snapshot():
    return {
        name: {"sha256": digest(p), "lines": len(p.read_bytes().splitlines())}
        for name, p in sources().items()
    }


def catalog(environment=None):
    def listing(prefix):
        names = sources() if environment is None else environment
        return {"modules": sorted(k for k in names if k.startswith(prefix.casefold()))}

    def count(module):
        # Lookup only the known top-level stdlib module names.
        table = sources() if environment is None else environment
        if module not in table:
            return {"error": "unknown module"}
        return {
            "lines": len(table[module].read_bytes().splitlines())
            if environment is None
            else table[module]["lines"]
        }

    def field(key):
        return {
            "type": "object",
            "properties": {key: {"type": "string"}},
            "required": [key],
            "additionalProperties": False,
        }

    return [
        Tool(
            "list_modules",
            "List installed top-level Python standard-library source modules starting with a prefix, in alphabetical order.",
            field("prefix"),
            listing,
        ),
        Tool(
            "module_lines",
            "Count source lines of the named installed Python standard-library module.",
            field("module"),
            count,
        ),
    ]
