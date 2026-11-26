"""Read individual records and values from CSV documents supplied by the caller."""

import csv
import os
from contextlib import contextmanager
from pathlib import Path

from .tool_session import Tool


def identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def tools(paths):
    documents = {}
    for path in map(Path, paths):
        if path.name in documents:
            raise ValueError(f"CSV documents need distinct file names: {path.name}")
        documents[path.name] = path, identity(path.stat())
    if not documents:
        raise ValueError("Supply at least one CSV document")

    @contextmanager
    def open_rows(document):
        if document not in documents:
            raise ValueError("Choose one of the supplied CSV documents")
        path, expected = documents[document]
        with path.open(encoding="utf-8-sig", newline="") as stream:
            if identity(os.fstat(stream.fileno())) != expected:
                raise ValueError("CSV document changed after the session opened")
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            if not header:
                raise ValueError("CSV document has no header")
            yield header, (record for record in reader if record)
            if identity(os.fstat(stream.fileno())) != expected:
                raise ValueError("CSV document changed while reading")

    def count(document):
        with open_rows(document) as (_, records):
            return {"value": sum(1 for _ in records)}

    def value(document, row, column):
        with open_rows(document) as (header, records):
            if header.count(column) != 1:
                raise ValueError("Column name must identify exactly one CSV field")
            if type(row) is not int or row < 1:
                raise ValueError("CSV data rows start at one")
            index = header.index(column)
            for number, record in enumerate(records, 1):
                if number == row:
                    if len(record) != len(header):
                        raise ValueError(
                            "The selected row and header have different widths"
                        )
                    return {"value": record[index]}
            raise ValueError("CSV row lies outside the document")

    name = {"type": "string", "enum": list(documents)}

    def schema(properties):
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    return [
        Tool(
            "csv_row_count",
            "Count nonempty data records after the header in a CSV document.",
            schema({"document": name}),
            count,
        ),
        Tool(
            "csv_cell",
            "Read the text in one CSV field. Data rows start at one, after the header.",
            schema(
                {
                    "document": name,
                    "row": {"type": "integer", "minimum": 1},
                    "column": {"type": "string", "description": "Exact column header."},
                }
            ),
            value,
        ),
    ]
