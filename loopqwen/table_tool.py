"""A read-only aggregate and lookup tool for an existing SQLite table."""

from .tool_session import Tool

OPERATIONS = {
    "value": None,
    "minimum": "MIN",
    "maximum": "MAX",
    "sum": "SUM",
    "average": "AVG",
    "count": "COUNT",
}
PREDICATES = {"equals": "=", "greater_than": ">", "less_than": "<"}


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def definition(columns):
    field = {"type": "string", "enum": list(columns)}
    where = {
        "type": "object",
        "properties": {
            "column": field,
            "operator": {"type": "string", "enum": list(PREDICATES)},
            "value": {"type": ["number", "string"]},
        },
        "required": ["column", "operator", "value"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "column": field,
            "operation": {
                "type": "string",
                "enum": list(OPERATIONS),
                "description": "Read matching values or aggregate the selected column.",
            },
            "filters": {
                "type": "array",
                "items": where,
                "description": "Conditions that must all match. Omit to read all rows.",
            },
        },
        "required": ["column", "operation"],
        "additionalProperties": False,
    }


def select(connection, table, columns, column, operation, filters=()):
    if column not in columns:
        raise ValueError("Unknown result column")
    projection = quote(column)
    aggregate = OPERATIONS[operation]
    if aggregate:
        projection = f"{aggregate}({projection})"
    clauses, parameters = [], []
    for cond in filters:
        key = cond["column"]
        if key not in columns:
            raise ValueError("Unknown filter column")
        clauses.append(f"{quote(key)} COLLATE NOCASE {PREDICATES[cond['operator']]} ?")
        parameters.append(cond["value"])
    predicate = " WHERE " + " AND ".join(clauses) if clauses else ""
    sql = f"SELECT {projection} FROM {quote(table)}{predicate} LIMIT 65"
    values = [r[0] for r in connection.execute(sql, parameters)]
    return {"values": values[:64], "truncated": len(values) > 64}


def database_tool(connection, table):
    """Expose one caller-selected table. The caller owns the connection."""
    columns = [
        r[0]
        for r in connection.execute(
            "SELECT name FROM pragma_table_info(?) ORDER BY cid", (table,)
        )
    ]
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Expected a visible SQLite table with distinct columns")

    def query(column, operation, filters=()):
        return select(connection, table, columns, column, operation, filters)

    return Tool(
        "query_table",
        "Read values or compute an aggregate over matching rows in " + table + ".",
        definition(columns),
        query,
    )
