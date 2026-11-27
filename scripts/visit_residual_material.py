"""Bind original qualitative question pairs, real requirement reversals and earlier genuine training turns."""

import re
from pathlib import Path

from loopqwen import qwen_tools
from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.tool_session import compact
from loopqwen.wheel_tools import tools
from scripts.followup_material import answer
from scripts.wheel_archive_sources import CACHE

TRAIN = Path("research/data/visit-residual-training.json")
PANEL = Path("research/data/visit-residual-panel.json")


def effects(w):
    a, b = w["questions"]
    for q, other in ((a, b), (b, a)):
        question = q["question"]
        text = f"Background\n{w['background']}\n\nSituation\n{w['situation']}\n\nAnswer the following question with a short answer.\n{question}"
        targets = [v["text"] for v in q["answers"]]
        yield {
            "id": "effects-" + q["id"],
            "family": "effects",
            "cluster": w["cluster"],
            "source": w["cluster"],
            "questions": [text],
            "answers": targets,
            "foil": other["answers"][0]["text"],
            "tools": [],
            "situation": w["situation"],
        }


def practical(w, other, release, allowed, cluster, family):
    names = sorted([w["file"], other["file"]])
    catalog = tools([CACHE / "wheels" / n for n in names])
    args = {"wheel": w["file"], "field": "Requires-Dist"}
    event = {
        "call": {"name": "wheel_metadata", "arguments": args},
        "observation": catalog[0].callback(**args),
    }
    prompt = (
        f"I am checking the published {w['package']} wheel, {w['file']}. "
        f"Read its declared direct dependency on {w['dependency']}. I may use {w['dependency']} {release[0]}. "
        "Give the exact version constraint on that dependency and say yes or no: does it allow this version?"
    )
    return {
        "id": family
        + "-"
        + fingerprint([w["file"], other["file"], release[0], cluster])[:16],
        "family": family,
        "cluster": cluster,
        "source": w["file"],
        "wheels": names,
        "tools": qwen_tools.declarations(catalog),
        "questions": [prompt],
        "current": event,
        "requirement": w["requirement"],
        "dependency": w["dependency"],
        "released_version": release[0],
        "allowed": allowed,
        "foil_wheel": other["file"],
    }


def training_views(w):
    messages = [{"role": "user", "content": w["questions"][0]}]
    common = {"group": w["family"], "source": w["source"], "tools": w["tools"]}
    if w["family"] == "effects":
        goal = w["answers"][0]
        sentence = next(
            (
                v
                for v in re.split(r"(?<=[.!?])\s+", w["situation"])
                if goal.casefold() in v.casefold()
            ),
            "",
        )
        reasoning = "The situation says: " + sentence if sentence else ""
        target = answer(goal, reasoning)
        alt = answer(w["foil"], reasoning)
    else:
        event = w["current"]
        call = event["call"]
        target = qwen_tools.assistant_message([call])
        yield {
            **common,
            "id": w["id"] + "-call",
            "messages": messages,
            "target": target,
            "alternative": qwen_tools.assistant_message(
                [{**call, "arguments": {**call["arguments"], "wheel": w["foil_wheel"]}}]
            ),
        }
        messages = [
            *messages,
            target,
            {"role": "tool", "content": compact(event["observation"])},
        ]
        yn = "yes" if w["allowed"] else "no"
        reasoning = (
            f"The declared requirement on {w['dependency']} is {w['requirement']}. {w['released_version']} "
            + ("satisfies it." if w["allowed"] else "falls outside it.")
        )
        target = answer(w["requirement"] + " | " + yn, reasoning)
        alt = answer(
            w["requirement"] + " | " + ("no" if w["allowed"] else "yes"), reasoning
        )
    yield {
        **common,
        "id": w["id"] + "-answer",
        "messages": messages,
        "target": target,
        "alternative": alt,
    }


def stored(path, data):
    if path.exists():
        assert read(path) == data
    else:
        save(path, data)


def build():
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    natural = read("research/sources/qualitative-effects/originals.json")
    packages = read("research/sources/dependency-decisions/originals.json")["cohorts"]
    swaps = read("research/sources/dependency-decisions/same-version.json")
    for split in ("train", "held"):
        rows = [
            r
            for w in natural["train" if split == "train" else "dev"]
            for r in effects(w)
        ]
        by_name = {w["file"]: w for w in packages[split]}
        for w in packages[split]:
            other = min(
                (v for v in packages[split] if v != w),
                key=lambda v: fingerprint([771930, w["file"], v["file"]]),
            )
            for key, allowed in (("allowed_release", True), ("denied_release", False)):
                rows.append(
                    practical(w, other, w[key], allowed, w["file"], "dependency")
                )
        for count, pair in enumerate(swaps[split]):
            a, b = [by_name[n] for n in pair["wheels"]]
            for idx, (w, other) in enumerate(((a, b), (b, a))):
                rows.append(
                    practical(
                        w,
                        other,
                        pair["release"],
                        pair["decisions"][idx],
                        f"swap-{count}",
                        "same_version",
                    )
                )
        if split == "train":
            fitted = [v for w in rows for v in training_views(w)]
            seen = {
                fingerprint([v["messages"], v["tools"], v["target"]]) for v in fitted
            }
            for name in (
                "native-function",
                "report-action",
                "cross-view",
                "proof-curriculum",
                "followup-protected",
            ):
                source = (
                    unpack(
                        Path(
                            "research/results/proof-curriculum-01/training-views.json.gz"
                        )
                    )
                    if name == "proof-curriculum"
                    else read("research/data/" + name + "-training.json")["examples"]
                )
                source = sorted(
                    source, key=lambda v: fingerprint([771931, name, v["id"]])
                )
                n = 0
                for w in source:
                    identity = fingerprint([w["messages"], w["tools"], w["target"]])
                    if identity in seen:
                        continue
                    seen.add(identity)
                    fitted.append(
                        {
                            "id": name + "/" + w["id"],
                            "source": w.get("source", w["id"]),
                            "group": name,
                            "messages": w["messages"],
                            "tools": w["tools"],
                            "target": w["target"],
                            "alternative": w.get("alternative"),
                        }
                    )
                    n += 1
                    if n == 320:
                        break
                assert n == 320
            assert len({v["id"] for v in fitted}) == len(fitted)
            stored(TRAIN, {"examples": fitted})
            print("Training turns", len(fitted))
        else:
            for w in rows:
                w["initial_ids"] = tok.encode(
                    qwen_tools.render(
                        tok,
                        [{"role": "user", "content": w["questions"][0]}],
                        w.pop("tools"),
                    ),
                    add_special_tokens=False,
                )
                for k in ("foil", "situation", "foil_wheel"):
                    w.pop(k, None)
            old = read("research/data/linked-rollin-panel.json")
            retained = [w for w in old["rows"] if "files" not in w]
            docs = {
                w["document"]: old["documents"][w["document"]]
                for w in retained
                if w["document"]
            }
            rows.extend(retained)
            rows.sort(key=lambda w: fingerprint([771932, w["id"]]))
            assert len(rows) == 112
            stored(PANEL, {"rows": rows, "documents": docs})
            print("Held requests", len(rows))


if __name__ == "__main__":
    build()
