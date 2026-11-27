"""Mount complete source tables and score the same proposed read on two human-labeled worlds."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from .. import qwen_tools
from ..relational_tools import tools
from ..tool_session import compact
from .native_function_study import executions

FILENAME = "table.sqlite"
SOURCE = Path("research/sources/changed-tables/originals.json")


def canonical(text):
    return " ".join(str(text).strip().casefold().split())


def present(answer, gold):
    """Accept the full denotation or a short answer sentence with a single isolated value."""
    if not answer:
        return False
    text = canonical(answer).replace("**", "").strip()
    value = canonical(gold)
    if text == value or text.rstrip(".!") == value:
        return True
    for prefix in ("the answer is ", "the answer was ", "answer: "):
        if (
            text.startswith(prefix)
            and text[len(prefix) :].strip().rstrip(".!") == value
        ):
            return True
    return False


def mount(folder, rows):
    folder.mkdir(exist_ok=True, parents=True)
    path = folder / FILENAME
    columns = rows[0]
    quoted = ['"' + header.replace('"', '""') + '" TEXT' for header in columns]
    # A distinct file is created per version. Values and column names are literal source text.
    if path.exists():
        raise FileExistsError(path)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE records (" + ", ".join(quoted) + ")")
        db.executemany(
            "INSERT INTO records VALUES (" + ",".join("?" for _ in columns) + ")",
            rows[1:],
        )
        assert [
            list(r) for r in db.execute("SELECT * FROM records ORDER BY rowid")
        ] == rows[1:]
    return path


def locations(folder, w):
    return [folder / w["id"] / str(v["world"]) / FILENAME for v in w["versions"]]


def prepare_tables(folder, cohort):
    for w in cohort:
        for path, v in zip(locations(folder, w), w["versions"], strict=True):
            mount(path.parent, v["rows"])


def prompt(w):
    return (
        "Use the supplied table to answer the question. Give the answer in a short phrase.\n"
        + w["question"]
    )


def state(w, folder):
    """Training-only schema-conditioned prompt. The same schema applies to both worlds."""
    paths = locations(folder, w)
    schema, query = tools([paths[0]])
    args = {"database_name": FILENAME}
    observed = schema.callback(**args)
    assert observed == tools([paths[1]])[0].callback(**args)
    messages = [
        {"role": "user", "content": prompt(w)},
        qwen_tools.assistant_message([{"name": schema.name, "arguments": args}]),
        {"role": "tool", "content": compact(observed)},
    ]
    return messages, qwen_tools.declarations([schema, query])


def query_results(statement, w, folder):
    result = []
    for path, v in zip(locations(folder, w), w["versions"], strict=True):
        provider = tools([path])[1]
        try:
            o = provider.callback(database_name=FILENAME, statement=statement)
            used = o["referenced_tables"] == ["records"] and not o["truncated"]
            denotation = bool(
                used
                and len(o["rows"]) == 1
                and len(o["rows"][0]) == 1
                and canonical(o["rows"][0][0]) == canonical(v["answer"])
            )
            result.append({"used": used, "correct": denotation, "observation": o})
        except (sqlite3.Error, ValueError, TypeError, OverflowError) as e:
            result.append({"used": False, "correct": False, "error": type(e).__name__})
    return result


def reward(visible, closed, definitions, w, folder):
    """The only positive signals are actual permitted reads and human-agreed denotations."""
    zero = {"reward": 0.0, "statement": None, "results": []}
    if not closed:
        return zero
    try:
        calls = qwen_tools.parse(
            visible, {v["function"]["name"]: v["function"] for v in definitions}
        )
    except (ValueError, TypeError, json.JSONDecodeError):
        return zero
    if len(calls) != 1 or calls[0]["name"] != "database_query":
        return zero
    args = calls[0]["arguments"]
    if (
        set(args) != {"database_name", "statement"}
        or args["database_name"] != FILENAME
        or not isinstance(args["statement"], str)
    ):
        return zero
    results = query_results(args["statement"], w, folder)
    valid = all(x["used"] for x in results)
    correct = sum(x["correct"] for x in results)
    return {
        "reward": 0.1 * valid + 0.2 * correct + 0.5 * (correct == 2),
        "statement": args["statement"],
        "results": results,
    }


def score(w, tr):
    read = False
    for event in executions(tr):
        o = event["observation"]
        if (
            event["call"]["name"] == "database_query"
            and o.get("referenced_tables") == ["records"]
            and o.get("rows")
        ):
            read = True
    finished = tr["status"] == "answered"
    ok = finished and present(tr["answer"], w["gold"])
    return {
        "completed": finished,
        "correct_answer": ok,
        "observed": read,
        "success": read and ok,
    }
