"""Freeze four source domains for one shared recurrent checkpoint."""

from collections import Counter
from pathlib import Path

from loopqwen import qwen_tools
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from scripts.dialogue_database_material import TRAIN as DB_TRAIN, PANEL as DB_PANEL
from scripts.human_assistance_sources import DEST as HUMAN

TRAIN = Path("research/data/integrated-assistance-training.json")
PANEL = Path("research/data/integrated-assistance-panel.json")


def build():
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    original = read(HUMAN / "originals.json")["splits"]
    old = Path("research/data")
    protected = read(old / "followup-protected-training.json")["examples"]
    states = read(old / "visit-residual-training.json")["examples"]
    previous = read(old / "followup-protected-panel.json")
    visits = read(old / "visit-residual-panel.json")
    assert previous["documents"] == visits["documents"]
    rows = list(visits["rows"])
    seen = {w["id"]: w for w in rows}
    for w in previous["rows"]:
        if w["id"] in seen:
            assert w == seen[w["id"]]
        else:
            rows.append(w)
    rows.extend(read(DB_PANEL)["rows"])
    train = []
    for group, items in (
        ("database", read(DB_TRAIN)["examples"]),
        (
            "reasoning",
            [w for w in states if w["group"] == "effects"]
            + [w for w in protected if w["kind"] in ("human_first", "human_followup")],
        ),
        (
            "practice",
            [w for w in protected if w["kind"] in ("action", "interpret")]
            + [w for w in states if w["group"] in ("dependency", "same_version")],
        ),
        ("rehearsal", [w for w in protected if w["kind"] == "protected"]),
    ):
        for w in items:
            train.append(
                {
                    "id": group + "/" + w["id"],
                    "group": group,
                    "kind": w.get("kind", w.get("group")),
                    **{k: w[k] for k in ("messages", "tools", "target")},
                }
            )
    for split, cohort in original.items():
        for w in cohort["selected"]:
            msgs = [{"role": "user", "content": w["question"]}]
            key = "oasst-" + w["tree"]
            if split == "train":
                train.append(
                    {
                        "id": key,
                        "group": "open",
                        "kind": "human_reply",
                        "messages": msgs,
                        "tools": [],
                        "target": {"role": "assistant", "content": w["answer"]},
                    }
                )
            else:
                rows.append(
                    {
                        "id": key,
                        "cluster": key,
                        "family": "open",
                        "source": w["tree"],
                        "questions": [w["question"]],
                        "tools": [],
                        "human_reply": w["answer"],
                        "initial_ids": tok.encode(
                            qwen_tools.render(tok, msgs, []), add_special_tokens=False
                        ),
                    }
                )
    assert len({r["id"] for r in rows}) == len(rows) == 220
    assert len({w["id"] for w in train}) == len(train)
    held = {tuple(r["initial_ids"]) for r in rows}
    # Check every training assistant turn, including observation-conditioned questions.
    unique_train = set()
    for w in train:
        prefix = tok.encode(
            qwen_tools.render(tok, w["messages"], w["tools"]), add_special_tokens=False
        )
        key = tuple(prefix)
        assert key not in held
        unique_train.add(fingerprint(prefix))
    rows.sort(key=lambda w: fingerprint([972361, w["id"]]))
    train.sort(key=lambda w: w["id"])
    counts = dict(Counter(w["group"] for w in train))
    summary = {
        "fits_by_group": counts,
        "distinct_train_openings": len(unique_train),
        "held_by_family": dict(Counter(w["family"] for w in rows)),
        "exact_train_held_prompt_overlaps": 0,
        "database_training": digest(DB_TRAIN),
        "database_panel": digest(DB_PANEL),
        "open_source": digest(HUMAN / "originals.json"),
    }
    for p, obj in (
        (TRAIN, {"examples": train, "source": summary}),
        (PANEL, {"rows": rows, "documents": visits["documents"], "source": summary}),
    ):
        if p.is_file():
            assert read(p) == obj
        else:
            save(p, obj)
    print(summary)


if __name__ == "__main__":
    build()
