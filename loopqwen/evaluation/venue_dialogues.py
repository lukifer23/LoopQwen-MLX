"""Execute place searches from human conversations against two actual SQLite inventories."""

from . import executable_tools, table_questions
from .answers import numeric

COLUMNS = [
    "name",
    "service",
    "area",
    "type",
    "pricerange",
    "food",
    "stars",
    "parking",
    "internet",
]
INSTRUCTION = "Use the table to count the venues that match the visitor’s current search, including all their venue preferences. Give the count only. The service column identifies hotel, restaurant or attraction. Use the name column to count venues."


def prompt_for(turns):
    return (
        "\n".join(
            ("Visitor" if t["speaker"] == "USER" else "Agent") + ": " + t["utterance"]
            for t in turns
        )
        + "\n\n"
        + INSTRUCTION
    )


def table(records):
    return dict(columns=COLUMNS, types=["text"] * len(COLUMNS), records=records)


def arguments(state):
    return dict(
        column="name",
        operation="count",
        filters=[
            dict(column=k, operator="equals", value=v) for k, v in sorted(state.items())
        ],
    )


def count(records, state):
    indices = [(COLUMNS.index(k), v) for k, v in state.items()]
    return sum(all(row[i] == v for i, v in indices) for row in records)


def workload(material):
    items = []
    for row in material["rows"]:
        if row.get("provider") == "bfcl_exec":
            items.append(
                dict(
                    row,
                    user_request=row["question"][0][0]["content"]
                    + executable_tools.INSTRUCTION,
                    generation_cap=384,
                )
            )
        else:
            items.append(dict(row, user_request=prompt_for(row["turns"])))

    def catalog(row):
        if row.get("provider") == "bfcl_exec":
            return executable_tools.tools_for(row)
        return table_questions.tools_for(
            table(material["inventories"][row["inventory"]])
        )

    def score(row, trace):
        if row.get("provider") == "bfcl_exec":
            return executable_tools.score(row, trace)
        gold = count(material["inventories"][row["inventory"]], row["state"])
        if row["gold"] != gold:
            raise ValueError("Frozen venue inventory and query disagree")
        calls = [t for t in trace["turns"] if "observation" in t]
        complete = trace["status"] == "answered"
        correct = numeric(trace["answer"] or "") == str(gold)
        path = False
        if calls:
            act = calls[-1]["call"]["arguments"]
            expected = {(k, "equals", v) for k, v in row["state"].items()}
            actual = {
                (f["column"], f["operator"], str(f["value"]).casefold())
                for f in act.get("filters", [])
            }
            path = (
                act["operation"] == "count"
                and act["column"] == "name"
                and actual == expected
                and calls[-1]["observation"] == {"values": [gold], "truncated": False}
            )
        return dict(
            success=bool(correct and complete and path),
            path=path,
            correct_answer=correct,
            completed=complete,
        )

    return items, catalog, score


def paired_result(material, records):
    from ..evaluation.statistics import exact_mcnemar

    by_side = {}
    retained = {}
    for side, rows in records.items():
        groups = {}
        retained[side] = 0
        for r, w in zip(rows, material["rows"], strict=True):
            if w.get("provider") == "bfcl_exec":
                retained[side] += r["score"]["success"]
                continue
            g = groups.setdefault((w["dialogue"], w["service"]), {})
            g[w["inventory"]] = r["score"]["success"]
        if any(set(g) != {"full", "subset"} for g in groups.values()):
            raise ValueError("Incomplete two-inventory dialogue")
        by_side[side] = {k: all(g.values()) for k, g in groups.items()}
    a, b = by_side["candidate"], by_side["control"]
    wins = sum(a[k] and not b[k] for k in a)
    losses = sum(b[k] and not a[k] for k in a)
    counts = {
        side: {
            s: sum(v for (_, domain), v in groups.items() if domain == s)
            for s in ("hotel", "restaurant", "attraction")
        }
        for side, groups in by_side.items()
    }
    return dict(
        wins=wins,
        losses=losses,
        p=exact_mcnemar(wins, losses),
        both_by_service=counts,
        public_tools=retained,
    )
