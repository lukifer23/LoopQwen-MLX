"""Apply the frozen task scorers to the same completed thinking turn."""

from . import extractive
from .answers import numeric
from .relations import conclusion
from .statistics import exact_mcnemar


def assess(item, ids, text):
    visible = None
    if (
        ids
        and ids[-1] in (248044, 248046)
        and 248068 not in ids
        and ids.count(248069) == 1
    ):
        boundary = "<|im_end|>" if ids[-1] == 248046 else "<|endoftext|>"
        if (
            text.count("</think>") != 1
            or not text.endswith(boundary)
            or set(ids[:-1]) & {248044, 248046}
        ):
            raise ValueError("Invalid first-turn boundary")
        visible = text.removesuffix(boundary).split("</think>")[1]
    family = item["family"]
    if family == "reading":
        answer = extractive.answer(ids, text)
        exact, f1 = extractive.score(answer, item["aliases"])
    else:
        answer = (
            (conclusion(visible) if family == "entailment" else numeric(visible))
            if visible is not None
            else None
        )
        exact = answer == item["gold"]
        f1 = None
    return {
        "answer": answer,
        "correct": exact,
        "f1": f1,
        "complete": visible is not None,
    }


def summarize(panel, records):
    scores = {}
    families = ["all", *sorted({r["family"] for r in panel})]
    for name in families:
        ix = [j for j, r in enumerate(panel) if name == "all" or r["family"] == name]
        weights = {}
        for arm, rows in records.items():
            rs = [rows[i] for i in ix]
            weights[arm] = {
                "correct": sum(r["correct"] for r in rs),
                "seconds": sum(r["elapsed_s"] for r in rs),
                "tokens": sum(len(r["token_ids"]) for r in rs),
            }
        differences = [
            int(records["candidate"][i]["correct"]) - int(records["base"][i]["correct"])
            for i in ix
        ]
        wins, losses = differences.count(1), differences.count(-1)
        scores[name] = dict(
            **weights,
            n=len(ix),
            wins=wins,
            losses=losses,
            p=exact_mcnemar(wins, losses),
        )
    return scores
