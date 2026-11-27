"""Inspect a caller-selected SQLite database and read bounded results across its tables."""

import json
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from string import ascii_lowercase, ascii_uppercase

from .tool_session import Tool

READ_FUNCTIONS = frozenset(
    "abs avg ceil ceiling char coalesce concat concat_ws count date datetime floor "
    "glob group_concat hex ifnull iif instr julianday lag lead length like likelihood "
    "likely lower ltrim max min nullif printf rank dense_rank replace round row_number "
    "rtrim sign strftime substr substring sum time total trim typeof unicode unixepoch "
    "unlikely upper".split()
)
MAX_ROWS = 50
MAX_BYTES = 32768
SQLITE_CASE = str.maketrans(ascii_uppercase, ascii_lowercase)


@contextmanager
def database(path):
    """Open an existing resource read-only and bound the work of a model-proposed SELECT."""
    with closing(
        sqlite3.connect(path.resolve(strict=True).as_uri() + "?mode=ro", uri=True, timeout=1)
    ) as db:
        db.execute("PRAGMA trusted_schema=OFF")
        db.execute("PRAGMA query_only=ON")
        db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1024 * 1024)
        db.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 65536)
        db.execute("BEGIN")
        tables = sorted(
            r[1]
            for r in db.execute("PRAGMA main.table_list")
            if r[0] == "main" and r[2] == "table" and not r[1].startswith("sqlite_")
        )
        yield db, tables


def select(path, statement, *, max_rows=MAX_ROWS):
    """Read a bounded result and record every table SQLite resolves for the query."""
    with database(path) as (db, tables):
        allowed = {t.translate(SQLITE_CASE): t for t in tables}
        referenced = set()
        steps = 0

        def progress():
            nonlocal steps
            steps += 1000
            return steps > 2_000_000

        def authorize(action, first, second, catalog, origin):
            if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
                return sqlite3.SQLITE_OK
            if (
                action == sqlite3.SQLITE_READ
                and catalog in (None, "main")
                and first is not None
                and first.translate(SQLITE_CASE) in allowed
            ):
                referenced.add(allowed[first.translate(SQLITE_CASE)])
                return sqlite3.SQLITE_OK
            if (
                action == sqlite3.SQLITE_FUNCTION
                and (second or "").lower() in READ_FUNCTIONS
            ):
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        db.set_authorizer(authorize)
        db.set_progress_handler(progress, 1000)
        cursor = db.execute(statement)
        if cursor.description is None:
            raise ValueError("A row-producing query is required")
        rows = []
        size = 0
        truncated = False
        for row in cursor:
            if len(rows) == max_rows:
                truncated = True
                break
            values = [
                {"hex": val.hex()} if isinstance(val, bytes) else val for val in row
            ]
            size += len(
                json.dumps(values, ensure_ascii=False, allow_nan=False).encode()
            )
            if size > MAX_BYTES:
                raise ValueError("Select fewer or shorter values from the database")
            rows.append(values)
        return {
            "columns": [c[0] for c in cursor.description],
            "rows": rows,
            "truncated": truncated,
            "referenced_tables": sorted(referenced),
        }


def tools(paths):
    files = {}
    for supplied in paths:
        path = Path(supplied).resolve(strict=True)
        if path.name in files and files[path.name] != path:
            raise ValueError(f"Two supplied databases share the name {path.name}")
        files[path.name] = path
    if not files:
        raise ValueError("Supply an existing SQLite database")
    names = {"type": "string", "enum": sorted(files)}

    def schema(database_name):
        with database(files[database_name]) as (db, tables):
            result = {}
            for name in tables:
                # Bind the table name as data. These introspections are fixed by the provider.
                cols = db.execute(
                    "SELECT name,type,pk FROM pragma_table_info(?) ORDER BY cid",
                    (name,),
                )
                foreign = db.execute(
                    'SELECT "from","table","to" FROM pragma_foreign_key_list(?) ORDER BY id,seq',
                    (name,),
                )
                result[name] = {
                    "columns": [list(r) for r in cols],
                    "foreign_keys": [list(r) for r in foreign],
                }
            return {"tables": result}

    def query(database_name, statement):
        return select(files[database_name], statement)

    def arguments(fields):
        return {
            "type": "object",
            "properties": fields,
            "required": list(fields),
            "additionalProperties": False,
        }

    return [
        Tool(
            "database_schema",
            "Inspect table names, columns (name, declared type, primary-key order) and foreign keys (column, referenced table, referenced column) in one supplied SQLite database.",
            arguments({"database_name": names}),
            schema,
        ),
        Tool(
            "database_query",
            "Run one read-only SQLite SELECT across the supplied database tables. Returns column names and up to 50 rows. Inspect database_schema for the names and relationships.",
            arguments({"database_name": names, "statement": {"type": "string"}}),
            query,
        ),
    ]
