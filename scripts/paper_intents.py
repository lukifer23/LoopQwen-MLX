"""What the model called and what the user ultimately learned on each live resource."""

import csv
import json
from fractions import Fraction

import matplotlib.pyplot as plt

from loopqwen.evaluation.answers import numeric
from loopqwen.evaluation.mechanism import unpack
from loopqwen.evaluation.paired_tool_intents import same_resource
from loopqwen.io import digest, read


def make(paper, results, result, finish):
    name = "tool-dialogue-intent-pairs-01"
    root = results / name
    if not (root / "checksums.json").exists():
        return
    decision = result(name)
    items = read(read(root / "registration.json")["material"])["rows"]
    checks = read(root / "checksums.json")
    with (paper / "intent-arithmetic-actions.csv").open("w") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "id",
                "source",
                "model",
                "question",
                "gold",
                "programs",
                "returns",
                "answer",
                "last_calculation",
                "echoed_exactly",
                "correct_final",
            ]
        )
        for side in ("candidate", "control"):
            path = root / (side + ".json.gz")
            assert digest(path) == checks[path.name]
            for w, r in zip(items, unpack(path), strict=True):
                if w["provider"] != "arithmetic":
                    continue
                tr = r["trajectory"]
                calls = [t for t in tr["turns"] if "observation" in t]
                obs = [t["observation"] for t in calls]
                last = obs[-1] if obs else {}
                value = Fraction(last["value"]) if "value" in last else None
                answer = numeric(tr["answer"] or "")
                state = "no callback"
                if obs:
                    state = (
                        "callback error"
                        if value is None
                        else (
                            "correct value"
                            if value == Fraction(w["gold"])
                            else "wrong value"
                        )
                    )
                writer.writerow(
                    [
                        r["id"],
                        w["source"],
                        side,
                        w["question"],
                        w["gold"],
                        json.dumps([t["call"] for t in calls]),
                        json.dumps(obs),
                        tr["answer"],
                        state,
                        value is not None
                        and answer is not None
                        and Fraction(answer) == value,
                        r["score"]["correct_answer"],
                    ]
                )
    totals = decision["resource_intents"]
    with (paper / "resource-actions.csv").open("w") as out:
        w = csv.writer(out)
        w.writerow(
            [
                "id",
                "shared_resource",
                "provider",
                "opaque",
                "requested_property",
                "name_order",
                "gold_arguments",
                "gold_value",
                "model",
                "calls",
                "observations",
                "correct_first_action",
                "correct_final",
                "strict",
                "status",
                "seconds",
            ]
        )
        for side in ("candidate", "control"):
            f = root / (side + ".json.gz")
            assert digest(f) == checks[f.name]
            for item, r in zip(items, unpack(f), strict=True):
                if item["provider"] != "resource":
                    continue
                assert r["id"] == item["id"]
                traj = r["trajectory"]
                turns = [v for v in traj["turns"] if "observation" in v]
                gold = item["names"][item["index"]]
                first = turns and turns[0]["call"]
                right = bool(
                    first
                    and first["name"] == gold
                    and same_resource(item["family"], first["arguments"], item["args"])
                )
                w.writerow(
                    [
                        r["id"],
                        item["pair"],
                        item["family"],
                        item["opaque"],
                        gold,
                        json.dumps([item["names"][j] for j in item["order"]]),
                        json.dumps(item["args"], sort_keys=True),
                        item["gold"],
                        side,
                        json.dumps([v["call"] for v in turns], sort_keys=True),
                        json.dumps([v["observation"] for v in turns], sort_keys=True),
                        right,
                        r["score"]["correct_answer"],
                        r["score"]["success"],
                        traj["status"],
                        traj["seconds"],
                    ]
                )
    fig, (act, ans) = plt.subplots(1, 2, figsize=(7.6, 3), sharey=True)
    families = list(totals["groups"]["candidate"])
    names = {
        "bfcl_exec": "Public numerical calls",
        "calendar": "Calendar",
        "zone": "Time zone",
        "distribution": "Packages",
        "unicode": "Unicode",
    }
    with (paper / "intent-scores.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["family", "side", "n", "strict", "correct_final", "seconds"])
        for i, family in enumerate(families):
            for side, offset, color, marker, label in (
                ("candidate", -0.11, "#11675d", "o", "Two passes"),
                ("control", 0.11, "#9d5932", "s", "Base"),
            ):
                cell = totals["groups"][side][family]
                w.writerow(
                    [
                        family,
                        side,
                        cell["n"],
                        cell["strict"],
                        cell["answer"],
                        cell["seconds"],
                    ]
                )
                for ax, k in ((act, "strict"), (ans, "answer")):
                    ax.scatter(
                        100 * cell[k] / cell["n"],
                        i + offset,
                        color=color,
                        marker=marker,
                        s=30,
                        label=label if i == 0 else None,
                    )
    act.set_yticks(range(len(families)), [names.get(k, k) for k in families])
    act.invert_yaxis()
    act.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=2, frameon=False)
    for ax, title in ((act, "Grounded success (%)"), (ans, "Correct final answer (%)")):
        ax.set_xlabel(title)
        ax.set_xlim(-4, 104)
        ax.set_xticks([0, 25, 50, 75, 100])
    finish(fig, "intent-performance")
