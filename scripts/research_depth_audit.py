"""Recompute paired depth-study measurements from the archived token streams."""

import re
from pathlib import Path

from loopqwen.evaluation.statistics import exact_mcnemar


def verify(root, decision, read, sha, require, number):
    reg = read(root / "registration.json")
    short = reg["id"] == "short-depth-transfer-01"
    panel = Path(
        reg["panel"]
        if short
        else reg.get("material", "research/data/deduction-stories-01.json")
    )
    require(
        sha(panel) == decision["data"]
        and sha(root / "registration.json") == decision["registration"],
        "Unbound depth evidence",
    )
    if decision["status"] == "interrupted":
        require(
            not decision["passed"] and bool(decision["reason"]),
            "Interrupted depth study promoted",
        )
        return
    data = read(panel)
    if not short:
        stories = [r["story"] for group in data.values() for r in group]
        require(
            len(stories) == len(set(stories)),
            "Repeated story across losses or evaluation",
        )
        data = data[reg["panel"]]
    scores = {}
    arms = ("recurrent", "dense") if short else reg["arms"]
    outputs = {}
    for arm in arms:
        archive = read(root / (arm + ".json.gz"))
        rows = archive["records"]
        outputs[arm] = rows
        require(len(rows) == len(data), "Incomplete paired study")
        for r, item in zip(rows, data, strict=True):
            require(
                r["id"] == item["id"]
                and r["prompt_ids"] == item["prompt_ids"]
                and 0 < len(r["token_ids"]) <= reg["generation_cap"],
                "Changed prompts or budget",
            )
            tokens = r["token_ids"]
            complete = (
                tokens[-1] in (248044, 248046)
                and tokens.count(248069) == 1
                and 248068 not in tokens
            )
            require(complete == r["complete"], "Incorrect first-turn completion")
            if short:
                pred = number(tokens, r["text"])
                gold = item["gold"]
            else:
                from loopqwen.research.deduction_data import answer

                body = re.split(r"<\|(?:im_end|endoftext)\|>$", r["text"])[0]
                pred = answer(body.split("</think>")[1]) if complete else None
                gold = item["answer"]
                words = ("cat", "mouse", "wolf", "sheep")
                require(
                    r["probe"]
                    == words[max(range(4), key=r["probe_logits"].__getitem__)],
                    "Wrong probe",
                )
                require(
                    r["end_to_end"] >= r["seconds"] > 0,
                    "Invalid actual generation time",
                )
            require(
                pred == r["answer"] and r["correct"] == (pred == gold),
                "Wrong graded answer",
            )
        scores[arm] = sum(r["correct"] for r in rows)
        if short:
            require(
                archive["weights"] == sha(Path(reg[arm]) / "adapters.safetensors"),
                "Changed trained lineage",
            )
        elif reg["arms"][arm].get("fit"):
            model = read(root / "reader.json")
            require(
                model["weights"] == sha(root / "reader.safetensors")
                and model["updates"] == reg["updates"],
                "Wrong trained reader",
            )

    def paired(group, first, second):
        diff = [
            int(a["correct"]) - int(b["correct"])
            for a, b, i in zip(outputs[first], outputs[second], data, strict=True)
            if group == "all" or i.get("source") == group
        ]
        return diff.count(1), diff.count(-1)

    a, b = arms
    wins, losses = paired("all", a, b)
    if short:
        for group, result in decision["scores"].items():
            require(
                (result["wins"], result["losses"]) == paired(group, a, b)
                and result["p"] == exact_mcnemar(result["wins"], result["losses"]),
                "Unpaired arithmetic statistics",
            )
            for arm in arms:
                subset = [
                    r
                    for r, i in zip(outputs[arm], data, strict=True)
                    if group == "all" or i["source"] == group
                ]
                require(
                    result[arm]["correct"] == sum(r["correct"] for r in subset)
                    and result[arm]["seconds"] == sum(r["seconds"] for r in subset),
                    "Arithmetic count or time",
                )
        v = decision["scores"]["all"]
        passed = (
            all(
                x["wins"] - x["losses"] >= 2
                for g, x in decision["scores"].items()
                if g != "all"
            )
            and v["p"] < 0.05
            and v[a]["seconds"] <= 1.3 * v[b]["seconds"]
            and decision["retention"][a] <= decision["retention"][b] + 0.03
        )
    else:
        for arm in arms:
            rows = outputs[arm]
            expected = dict(
                correct=scores[arm],
                complete=sum(r["complete"] for r in rows),
                probe=sum(
                    r["probe"] == x["answer"] for r, x in zip(rows, data, strict=True)
                ),
                seconds=sum(r["end_to_end"] for r in rows),
                tokens=sum(len(r["token_ids"]) for r in rows),
            )
            require(decision["scores"][arm] == expected, "Invalid deduction summary")
        u, v = decision["scores"][a], decision["scores"][b]
        require(
            (wins, losses) == (decision["wins"], decision["losses"])
            and decision["p"] == exact_mcnemar(wins, losses),
            "Invalid paired deduction decision",
        )
        passed = (
            wins - losses >= reg["net_floor"]
            and decision["p"] < 0.05
            and u["probe"] >= v["probe"] + reg["probe_floor"]
            and u["seconds"] <= reg["time_ratio"] * v["seconds"]
        )
    require(passed == decision["passed"], "Incorrect promotion decision")
