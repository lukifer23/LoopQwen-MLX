"""Fit on publisher training queries and reserve previously unused evaluation tables."""

import random
from pathlib import Path

from loopqwen.evaluation import table_questions as sql
from loopqwen.io import read, save
from loopqwen.tool_session import compact, initial_messages
from table_question_panel import REV, select

DATA = Path("research/data")
OP = ["value", "maximum", "minimum", "count", "sum", "average"]
CMP = ["equals", "greater_than", "less_than"]


def main():
    old = read(DATA / "tool-table-queries-01.json")["rows"]
    exclude = {r["table_id"] for r in old}
    questions = {r["question"].casefold().strip() for r in old}
    training = select("train", 128, "sqlite-fit-31950", exclude, questions)
    exclude.update(r["table_id"] for r in training)
    questions.update(r["question"].casefold().strip() for r in training)
    test = select("test", 6, "sqlite-blind-15587", exclude, questions)
    examples = []
    for r in training:
        t = sql.tools_for(r)[0]
        params = r["sql"]
        cols = r["columns"]
        args = dict(
            column=cols[params["sel"]],
            operation=OP[params["agg"]],
            filters=[
                dict(column=cols[i], operator=CMP[op], value=v)
                for i, op, v in params["conds"]
            ],
        )
        action = compact(dict(name=t.name, arguments=args))
        obs = t.callback(**args)
        ref = sql.reference(r)
        if obs["values"] != ref:
            raise ValueError(
                "Training action does not return publisher denotation: " + r["id"]
            )
        answer = format(ref[0], ".13g")
        messages = initial_messages(r["question"] + sql.INSTRUCTION, [t])
        examples.append(
            dict(id=r["id"] + "-call", messages=messages, thinking="", visible=action)
        )
        examples.append(
            dict(
                id=r["id"] + "-answer",
                messages=[
                    *messages,
                    dict(role="assistant", content=action),
                    dict(role="tool", content=compact(obs)),
                ],
                thinking="",
                visible=answer,
            )
        )
    examples.extend(
        random.Random(98525).sample(
            read(DATA / "tool-dialogue-typed-training.json")["examples"], 512
        )
    )
    save(
        DATA / "tool-sql-learning-training.json",
        dict(
            source=REV,
            training_tables=[r["table_id"] for r in training],
            examples=examples,
        ),
    )
    rows = [dict(r, provider="wikisql") for r in test]
    rows.extend(
        dict(r, provider="bfcl_exec")
        for r in read(DATA / "tool-exec-bfcl-01.json")["rows"]
    )
    random.Random(13797).shuffle(rows)
    save(DATA / "tool-sql-learning-panel.json", dict(source=REV, rows=rows))
    print(len(training), len(test), len(examples))


if __name__ == "__main__":
    main()
