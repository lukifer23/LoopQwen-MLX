"""Read the original core-metadata headers of wheel archives supplied by the caller."""

import re
from email import policy
from email.parser import BytesHeaderParser
from pathlib import Path
from zipfile import ZipFile

from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

from .tool_session import Tool

HEADER = r"^[A-Za-z][A-Za-z0-9-]*$"


def headers(path):
    name, version, _, _ = parse_wheel_filename(path.name)
    with ZipFile(path) as archive:
        members = [
            member
            for member in archive.infolist()
            if member.filename.count("/") == 1
            and member.filename.endswith(".dist-info/METADATA")
        ]
        if len(members) != 1:
            raise ValueError(
                f"Expected one top-level distribution METADATA file in {path.name}"
            )
        with archive.open(members[0]) as stream:
            header_bytes = bytearray()
            for line in stream:
                if line in (b"\n", b"\r\n"):
                    break
                header_bytes.extend(line)
            message = BytesHeaderParser(policy=policy.default).parsebytes(
                bytes(header_bytes)
            )
    names, versions = message.get_all("Name", []), message.get_all("Version", [])
    if (
        len(names) != 1
        or len(versions) != 1
        or canonicalize_name(str(names[0]), validate=True) != name
        or Version(str(versions[0])) != version
    ):
        raise ValueError(f"Embedded distribution identity differs from {path.name}")
    return message


def tools(paths):
    snapshots = {}
    for entry in paths:
        path = Path(entry).resolve(strict=True)
        if path.name in snapshots:
            if path != snapshots[path.name][0]:
                raise ValueError(f"Two supplied wheels share the name {path.name}")
            continue
        snapshots[path.name] = path, headers(path)
    if not snapshots:
        raise ValueError("Supply at least one wheel archive")

    def read_header(wheel, field):
        if not re.fullmatch(HEADER, field):
            raise ValueError("Invalid metadata header name")
        return {"values": [str(v) for v in snapshots[wheel][1].get_all(field, [])]}

    return [
        Tool(
            "wheel_metadata",
            "Read all values of one core-metadata header from a supplied Python wheel. An absent header returns an empty list.",
            {
                "type": "object",
                "properties": {
                    "wheel": {"type": "string", "enum": list(snapshots)},
                    "field": {
                        "type": "string",
                        "pattern": HEADER,
                        "description": "Core metadata header name, such as Name, Version, Summary, Requires-Dist or Requires-Python.",
                    },
                },
                "required": ["wheel", "field"],
                "additionalProperties": False,
            },
            read_header,
        )
    ]
