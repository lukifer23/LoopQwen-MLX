"""Word-aligned literal and header pointers for a real query_table call."""

import re

OPERATIONS = ("value", "maximum", "minimum", "count", "sum", "average")
COMPARISONS = ("equals", "greater_than", "less_than")
CONDITIONS = 4
SPAN = 12
WORD = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+(?:['’][^\W\d_]+)*|[^\w\s]", re.UNICODE)


def words(question):
    return [(m.start(), m.end()) for m in WORD.finditer(question)]


def normalized(value):
    return "".join(c for c in str(value).casefold() if c.isalnum())


def actions(row):
    q = row["question"]
    spans = words(q)
    literals = {}
    for i, (a, _) in enumerate(spans):
        for j in range(i, min(len(spans), i + SPAN)):
            b = spans[j][1]
            value = q[a:b]
            key = normalized(value)
            if (
                key
                and key not in literals
                and value[0].isalnum()
                and value[-1].isalnum()
            ):
                literals[key] = (i, j)
    conditions = []
    for col, op, value in row["sql"]["conds"]:
        if op >= len(COMPARISONS) or normalized(value) not in literals:
            return None
        start, end = literals[normalized(value)]
        conditions.append([col, op, start, end])
    if len(conditions) > CONDITIONS:
        return None
    return sorted(conditions, key=lambda c: (c[2], c[3], c[0], c[1]))


def arguments(question, columns, select, aggregate, conditions):
    positions = words(question)
    filters = []
    for col, op, left, right in conditions:
        a, b = positions[left][0], positions[right][1]
        literal = question[a:b]
        if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", literal):
            literal = literal.replace(",", "")
        filters.append(
            {"column": columns[col], "operator": COMPARISONS[op], "value": literal}
        )
    return {
        "column": columns[select],
        "operation": OPERATIONS[aggregate],
        "filters": filters,
    }
