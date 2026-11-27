"""Read the byte size and logical line count of files selected by the caller."""

import os
import stat
from pathlib import Path

from .tool_session import Tool


def definitions(lookup, paths=None):
    path_schema = {"type": "string"}
    if paths is not None:
        path_schema["enum"] = list(paths)
    schema = {
        "type": "object",
        "properties": {"path": path_schema},
        "required": ["path"],
        "additionalProperties": False,
    }
    location = (
        "project file by its exact repository-relative path"
        if paths is None
        else "file by its supplied path"
    )
    return [
        Tool(
            "file_bytes",
            f"Get the size in bytes of a {location}.",
            schema,
            lambda path: lookup(path, "bytes"),
        ),
        Tool(
            "file_lines",
            f"Get the number of lines in a {location}.",
            schema,
            lambda path: lookup(path, "lines"),
        ),
    ]


def line_count(stream, block=1 << 20):
    """Match bytes.splitlines() for CR, LF and CRLF, including a split chunk boundary."""
    ends = 0
    tail = b""
    while chunk := stream.read(block):
        ends += chunk.count(b"\n") + chunk.count(b"\r") - chunk.count(b"\r\n")
        if tail == b"\r" and chunk.startswith(b"\n"):
            ends -= 1
        tail = chunk[-1:]
    return ends + int(tail not in (b"", b"\r", b"\n"))


def identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def tools(paths):
    snapshots = {}
    for name in dict.fromkeys(map(str, paths)):
        file = Path(name).resolve(strict=True)
        info = file.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Expected a regular file: " + name)
        snapshots[name] = (file, identity(info))
    if not snapshots:
        raise ValueError("Supply at least one file")
    counts = {}

    def lookup(path, field):
        if path not in snapshots:
            raise FileNotFoundError(path)
        file, original = snapshots[path]
        if identity(file.stat()) != original:
            raise ValueError("The selected file changed during this conversation")
        if field == "bytes":
            return {"bytes": original[2]}
        if path not in counts:
            with file.open("rb") as stream:
                if identity(os.fstat(stream.fileno())) != original:
                    raise ValueError("The selected file changed before reading")
                value = line_count(stream)
                if identity(os.fstat(stream.fileno())) != original:
                    raise ValueError("The selected file changed while reading")
            counts[path] = value
        return {"lines": counts[path]}

    return definitions(lookup, snapshots)
