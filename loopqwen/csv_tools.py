"""Read individual records and values from CSV documents supplied by the caller."""

import csv
import os
from contextlib import contextmanager
from pathlib import Path

from .tool_session import Tool


def identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def tools(paths, *, keyed=False):
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

    if keyed:
        headers = set()
        for document in documents:
            with open_rows(document) as (header, _):
                headers.update(header)
        column = {"type": "string", "enum": sorted(headers)}

        def lookup(document, key_column, key, value_column):
            with open_rows(document) as (header, records):
                if header.count(key_column) != 1 or header.count(value_column) != 1:
                    raise ValueError("Choose unambiguous columns from the document")
                search, target = header.index(key_column), header.index(value_column)
                value = None
                for record in records:
                    if len(record) != len(header):
                        raise ValueError("Row and header have different widths")
                    if record[search] == key:
                        if value is not None:
                            raise ValueError("Multiple CSV records match the key")
                        value = record[target]
                if value is None:
                    raise ValueError("No CSV record matches the key")
                return {"value": value}

        return [
            Tool(
                "csv_lookup",
                "Read a field from the one CSV row whose key column equals the supplied text. Key comparison is exact.",
                schema(
                    {
                        "document": name,
                        "key_column": column,
                        "key": {"type": "string"},
                        "value_column": column,
                    }
                ),
                lookup,
            )
        ]

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
