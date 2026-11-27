"""Freeze authentic two-question reading and changed-archive tasks at the deployed interface."""

from pathlib import Path

from loopqwen import qwen_tools
from loopqwen.identity import fingerprint
from loopqwen.io import read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.linked_rollin_data import protected_views
from loopqwen.tool_session import compact
from loopqwen.wheel_tools import tools
from scripts.followup_sources import DEST
from scripts.wheel_archive_sources import CACHE

TRAIN = Path("research/data/followup-protected-training.json")
PANEL = Path("research/data/followup-protected-panel.json")


def human_answer(q):
    a = q["teacher"]["text"]
    return "Cannot answer" if a == "CANNOTANSWER" else a


def answer(text, reason=""):
    return {"role": "assistant", "content": text, "reasoning_content": reason}


def dialogue(row):
    a, b = row["turns"]
    body = (
        f"Article: {row['title']}\nSection: {row['section']}\nBackground: {row['background']}\n\n"
        + row["passage"]
        + "\n\nAnswer the current question with an excerpt from the section. If the section does not answer it, say Cannot answer."
    )
    for i, (q, other) in enumerate(((a, b), (b, a))):
        prior = (
            ""
            if not i
            else f"\n\nPrevious question: {a['question']}\nPrevious answer: {human_answer(a)}"
        )
        user = body + prior + "\n\nCurrent question: " + q["question"]
        yield {
            "id": q["id"],
            "cluster": row["id"],
            "family": "human_first" if i == 0 else "human_followup",
            "questions": [user],
            "tools": [],
            "source": row["title"],
            "passage": row["passage"],
            "turn": i,
            "previous_answer": human_answer(a) if i else None,
            "annotations": [z["text"] for z in q["annotations"]],
            "target": human_answer(q),
            "foil": human_answer(other),
        }


def practical(row, peers):
    requirement = row["headers"]["Requires-Python"][0]
    choices = [w for w in peers if w["headers"]["Requires-Python"][0] != requirement]
    previous = min(choices, key=lambda w: fingerprint([740037, row["file"], w["file"]]))
    prior = previous["headers"]["Requires-Python"][0]
    filenames = sorted([row["file"], previous["file"]])
    catalog = tools([CACHE / "wheels" / name for name in filenames])
    defs = qwen_tools.declarations(catalog)

    def event(w):
        args = {"wheel": w["file"], "field": "Requires-Python"}
        return {
            "call": {"name": catalog[0].name, "arguments": args},
            "observation": catalog[0].callback(**args),
        }

    past, fresh = event(previous), event(row)
    assert past["observation"] == {"values": [prior]} and fresh["observation"] == {
        "values": [requirement]
    }
    for py in row["boundary"]:
        allowed = dict(row["interpreters"])[py]
        prior_allowed = dict(previous["interpreters"])[py]
        decision = "yes" if allowed else "no"
        prompt = (
            f"We previously checked {previous['package']} {previous['version']}. Its wheel declared {prior}. "
            f"I am now reviewing {row['package']} {row['version']} for Python {py}. "
            "Read the new wheel. Give its exact Requires-Python constraint and say yes or no: does the declared constraint allow this interpreter?"
        )
        yield {
            "id": row["file"] + "@" + py,
            "cluster": row["package"],
            "family": "archive",
            "questions": [prompt],
            "source": row["url"],
            "wheels": filenames,
            "tools": defs,
            "previous": past,
            "current": fresh,
            "interpreter": py,
            "requirement": requirement,
            "prior_requirement": prior,
            "allowed": allowed,
            "prior_allowed": prior_allowed,
            "target": requirement + " | " + decision,
            "foil": prior + " | " + ("yes" if prior_allowed else "no"),
        }


def examples(w):
    messages = [{"role": "user", "content": w["questions"][0]}]
    base = {"tools": w["tools"], "source": w["source"]}
    if w["family"] == "archive":
        action = qwen_tools.assistant_message([w["current"]["call"]])
        wrong = qwen_tools.assistant_message([w["previous"]["call"]])
        yield dict(
            id=w["id"] + "-open",
            kind="action",
            messages=messages,
            target=action,
            alternative=wrong,
            **base,
        )
        messages = [
            *messages,
            action,
            {"role": "tool", "content": compact(w["current"]["observation"])},
        ]
        reason = (
            f"The requested archive declares {w['requirement']}. Python {w['interpreter']} "
            + ("satisfies" if w["allowed"] else "does not satisfy")
            + " that constraint."
        )
        kind = "interpret"
    else:
        reason = (
            "The section does not provide the requested information."
            if w["target"] == "Cannot answer"
            else "The relevant excerpt is: " + w["target"]
        )
        kind = w["family"]
    # Compare the two answers after the same positive reasoning prefix; both visible texts came from real sources.
    yield dict(
        id=w["id"] + "-answer",
        kind=kind,
        messages=messages,
        target=answer(w["target"], reason),
        alternative=answer(w["foil"], reason) if w["target"] != w["foil"] else None,
        **base,
    )


def build():
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    sources = read(DEST / "originals.json")
    wheels = read("research/sources/wheel-archives/enrollment.json")["cohorts"]
    for split in ("train", "dev"):
        rows = []
        for w in sources[split]["rows"]:
            rows.extend(dialogue(w))
        peers = wheels["train" if split == "train" else "held"]
        for w in peers:
            rows.extend(practical(w, peers))
        if split == "train":
            views = [v for w in rows for v in examples(w)]
            views.extend(
                {**w, "kind": "protected", "alternative": None}
                for w in protected_views()
            )
            dest = TRAIN
            output = {"examples": views}
        else:
            for w in rows:
                w["initial_ids"] = tok.encode(
                    qwen_tools.render(
                        tok,
                        [{"role": "user", "content": w["questions"][0]}],
                        w["tools"],
                    ),
                    add_special_tokens=False,
                )
            prior = read("research/data/linked-rollin-panel.json")
            retained = [w for w in prior["rows"] if "files" not in w]
            assert len(retained) == 36
            documents = {
                w["document"]: prior["documents"][w["document"]]
                for w in retained
                if w["document"]
            }
            for w in rows:
                w.pop("foil")
                w.pop("tools")
            rows.extend(retained)
            rows.sort(key=lambda w: fingerprint([740044, w["id"]]))
            dest = PANEL
            output = {"rows": rows, "documents": documents}
        if dest.exists():
            assert read(dest) == output
        else:
            save(dest, output)
        print(split, len(views) if split == "train" else len(rows))


if __name__ == "__main__":
    build()
