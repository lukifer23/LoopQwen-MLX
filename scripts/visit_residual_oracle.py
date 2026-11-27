"""Independently open every held wheel, grade both actual release directions and check source isolation."""

from email.parser import BytesParser
from email import policy
from zipfile import ZipFile
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.specifiers import SpecifierSet

from loopqwen.evaluation.extractive import normalized
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from loopqwen.research.visit_residual_fit import annotations
from loopqwen.research.visit_residual_study import score, tools_for
from scripts.dependency_crossovers import build as swaps
from scripts.dependency_sources import enrolled
from scripts.qualitative_sources import enroll
from scripts.visit_residual_material import PANEL, TRAIN
from scripts.wheel_archive_sources import CACHE

RECEIPT = "docs/measurements/visit-residual-sources.json"


def audit():
    source = enroll()
    dep = enrolled()
    swaps()
    roots = {w["file"]: w for r in dep["cohorts"].values() for w in r}
    human = {q["id"]: q for w in source["dev"] for q in w["questions"]}
    train = read(TRAIN)["examples"]
    rows = read(PANEL)["rows"]
    prompts = {
        normalized(m["content"])
        for v in train
        for m in v["messages"]
        if m["role"] == "user"
    }
    concat = "\n".join(prompts)
    counts = {}
    n = 0
    for w in source["dev"]:
        assert all(normalized(w[k]) not in concat for k in ("background", "situation"))
    for w in rows:
        assert normalized(w["questions"][0]) not in prompts
        if w["family"] == "effects":
            original = human[w["id"].removeprefix("effects-")]
            assert w["questions"][0].endswith(original["question"])
            assert w["answers"] == [v["text"] for v in original["answers"]]
            for ans in w["answers"]:
                assert score(None, w, 0, {"status": "answered", "answer": ans})[
                    "success"
                ]
        if "wheels" in w:
            item = roots[w["source"]]
            file = CACHE / "wheels" / item["file"]
            assert digest(file) == item["sha256"]
            with ZipFile(file) as f:
                name = next(
                    n for n in f.namelist() if n.endswith(".dist-info/METADATA")
                )
                h = BytesParser(policy=policy.compat32).parsebytes(
                    f.read(name), headersonly=True
                )
            reqs = [Requirement(v) for v in h.get_all("Requires-Dist", [])]
            (required,) = [
                r
                for r in reqs
                if canonicalize_name(r.name) == w["dependency"] and r.marker is None
            ]
            assert required.specifier == SpecifierSet(w["requirement"])
            assert required.specifier.contains(w["released_version"]) == w["allowed"]
            (tool,) = tools_for(w, None)
            event = w["current"]
            assert event["observation"] == tool.callback(**event["call"]["arguments"])
            ans = w["requirement"] + " | " + ("yes" if w["allowed"] else "no")
            assert score(
                None, w, 0, {"status": "answered", "answer": ans, "turns": [event, {}]}
            )["success"]
            assert not score(
                None, w, 0, {"status": "answered", "answer": ans, "turns": [{}]}
            )["success"]
            peer = next(k for k in w["wheels"] if k != w["source"])
            args = {"wheel": peer, "field": "Requires-Dist"}
            false_event = {
                "call": {"name": "wheel_metadata", "arguments": args},
                "observation": tool.callback(**args),
            }
            assert not score(
                None,
                w,
                0,
                {"status": "answered", "answer": ans, "turns": [false_event, {}]},
            )["success"]
            if w["family"] == "same_version":
                counts.setdefault(w["cluster"], []).append(
                    (w["dependency"], w["released_version"], w["allowed"])
                )
            n += 1
    assert len(counts) == 4 and all(
        len(x) == 2 and x[0][:2] == x[1][:2] and x[0][2] != x[1][2]
        for x in counts.values()
    )
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    lengths = []
    targets = comp = 0
    for v in train:
        x, y = token_pair(tok, v)
        _, _, c = annotations(tok, v, y)
        comp += int(c is not None)
        targets += len(y)
        lengths.append(len(x) + len(y) - 1)
    receipt = {
        "training": digest(TRAIN),
        "panel": digest(PANEL),
        "views": len(train),
        "original_targets": targets,
        "visible_contrasts": comp,
        "max_feature_input": max(lengths),
        "original_human_held": len(human),
        "original_wheel_decisions": n,
        "same_release_pairs": len(counts),
        "exact_held_user_messages_in_training": 0,
    }
    from pathlib import Path

    if Path(RECEIPT).exists():
        assert read(RECEIPT) == receipt
    else:
        save(RECEIPT, receipt)
    print(receipt)
    return receipt


if __name__ == "__main__":
    audit()
