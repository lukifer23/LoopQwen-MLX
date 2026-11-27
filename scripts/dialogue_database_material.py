"""Bind authentic human questions, original expert SQL and live database returns to the public chat format."""

from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from loopqwen import qwen_tools
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.relational_tools import tools
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from loopqwen.tool_session import compact
from scripts.dialogue_database_sources import CACHE, DEST

TRAIN = Path("research/data/database-dialogue-training.json")
PANEL = Path("research/data/database-dialogue-panel.json")


def prompt(history, question):
    previous = "\n".join(
        ("Customer: " if msg["role"] == "user" else "Analyst: ") + msg["content"]
        for msg in history
    )
    transcript = ("Conversation so far:\n" + previous + "\n\n") if previous else ""
    return (
        transcript
        + "Use the supplied database to answer the customer's current question.\n"
        + question
    )


def build():
    source = read(DEST / "originals.json")
    assert digest(CACHE) == source["archive_sha256"]
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    training, panel = [], []
    train_user, held_user = set(), set()
    with ZipFile(CACHE) as archive, TemporaryDirectory() as tmp:
        for split, dbs in source["cohorts"].items():
            for db in dbs:
                filename = db["database"] + ".sqlite"
                path = Path(tmp) / filename
                path.write_bytes(archive.read(db["member"]))
                assert digest(path) == db["sqlite_sha256"]
                schema, query = tools([path])
                defs = qwen_tools.declarations((schema, query))
                schema_args = {"database_name": filename}
                shape = schema.callback(**schema_args)
                for dialog in db["dialogues"]:
                    history = list(dialog["prior"])
                    for i, q in enumerate(dialog["sequence"]):
                        text = prompt(history, q["question"])
                        identifier = (
                            "cosql-"
                            + fingerprint(
                                [db["database"], dialog["source"], dialog["turn"], i]
                            )[:20]
                        )
                        messages = [{"role": "user", "content": text}]
                        obs = query.callback(
                            database_name=filename, statement=q["query"]
                        )
                        assert obs["rows"] == [[q["returned"]]] and not obs["truncated"]
                        actions = [
                            (schema, schema_args, shape),
                            (
                                query,
                                {"database_name": filename, "statement": q["query"]},
                                obs,
                            ),
                        ]
                        common = {
                            "source": db["database"],
                            "dialogue": dialog["source"],
                            "tools": defs,
                        }
                        if split == "train":
                            train_user.add(text)
                            for tool, args, observation in actions:
                                target = qwen_tools.assistant_message(
                                    [{"name": tool.name, "arguments": args}]
                                )
                                training.append(
                                    {
                                        **common,
                                        "id": identifier + "-" + tool.name,
                                        "group": tool.name,
                                        "messages": list(messages),
                                        "target": target,
                                    }
                                )
                                messages.extend(
                                    (
                                        target,
                                        {
                                            "role": "tool",
                                            "content": compact(observation),
                                        },
                                    )
                                )
                            training.append(
                                {
                                    **common,
                                    "id": identifier + "-answer",
                                    "group": "database_answer",
                                    "messages": list(messages),
                                    "target": {
                                        "role": "assistant",
                                        "content": q["human_answer"],
                                    },
                                }
                            )
                        else:
                            held_user.add(text)
                            panel.append(
                                {
                                    **common,
                                    "family": "database",
                                    "cluster": db["database"],
                                    "id": identifier,
                                    "turn": i,
                                    "questions": [text],
                                    "sql": q["query"],
                                    "expected_value": q["returned"],
                                    "human_answer": q["human_answer"],
                                    "original_question": q["question"],
                                    "initial_ids": tok.encode(
                                        qwen_tools.render(tok, messages, defs),
                                        add_special_tokens=False,
                                    ),
                                    "sqlite_member": db["member"],
                                    "sqlite_sha256": db["sqlite_sha256"],
                                }
                            )
                        history.extend(
                            (
                                {"role": "user", "content": q["question"]},
                                {"role": "assistant", "content": q["human_answer"]},
                            )
                        )
                path.unlink()
    assert len(training) == 666 and len(panel) == 24
    overlap = train_user & held_user
    excluded = {
        w["dialogue"] for w in training if w["messages"][0]["content"] in overlap
    }
    # Drop the full training dialogue, including follow-ups that might quote the duplicate question.
    training = [w for w in training if w["dialogue"] not in excluded]
    train_user = {w["messages"][0]["content"] for w in training}
    assert not train_user & held_user and len(excluded) == 1 and len(training) == 660
    assert not {w["source"] for w in training} & {r["source"] for r in panel}
    lengths = [tuple(map(len, token_pair(tok, w))) for w in training]
    report = {
        "cohort_sha256": digest(DEST / "originals.json"),
        "train_views": len(training),
        "excluded_training_dialogues_for_exact_prompt_overlap": sorted(excluded),
        "train_questions": len(train_user),
        "held_questions": len(held_user),
        "train_tokens": sum(y for x, y in lengths),
        "max_training_sequence": max(x + y for x, y in lengths),
        "held_schemas": len(set(r["source"] for r in panel)),
    }
    for path, obj in (
        (TRAIN, {"provenance": report, "examples": training}),
        (PANEL, {"provenance": report, "rows": panel}),
    ):
        if path.exists():
            assert read(path) == obj
        else:
            save(path, obj)
    print(report, flush=True)


if __name__ == "__main__":
    build()
