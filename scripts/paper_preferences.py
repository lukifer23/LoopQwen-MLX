"""Plot both sides of file clarification and export every answer and actual action."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read
from loopqwen.tool_session import compact


def make(paper, results, result, finish):
    name = "action-preference-01"
    import json

    from paper_overlap import audit

    from loopqwen.tool_session import SYSTEM

    root = results / name
    spec = read(Path("research/experiments") / (name + ".json"))
    fit = read(spec["training"])
    material = read(spec["material"])["rows"]
    pref = [w for w in fit["preferences"] if w["split"] == "train"]
    training = [w["messages"][-1]["content"] for w in pref]
    training += [
        m["content"]
        for w in fit["replay"]
        for m in w["messages"]
        if m["role"] == "user"
    ]
    names = set()
    for w in [*pref, *fit["replay"]]:
        for m in w["messages"]:
            if m["role"] == "system" and m["content"].startswith(SYSTEM):
                names.update(v["name"] for v in json.loads(m["content"][len(SYSTEM) :]))
    with (paper / "action-preference-inputs.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "cohort",
                "numbers_masked",
                "unique_fit_questions",
                "unique_enrolled_questions",
                "exact_matches",
                "near_matches",
                "max_fivegram_jaccard",
                "csv_operation_names_seen_in_fit",
            ]
        )
        for cohort in (
            "initial file requests",
            "file follow-ups",
            "preference validation",
        ):
            if cohort == "preference validation":
                held = [
                    w["messages"][-1]["content"]
                    for w in fit["preferences"]
                    if w["split"] == "validation"
                ]
            elif cohort == "file follow-ups":
                held = [w["users"][1] for w in material if w["kind"] == "missing"]
            else:
                held = [w["users"][0] for w in material if w["kind"] != "public"]
            for mask in (False, True):
                writer.writerow(
                    [
                        cohort,
                        mask,
                        *audit(training, held, mask),
                        len(names & {"csv_cell", "csv_row_count"}),
                    ]
                )
    if not (root / "checksums.json").is_file():
        return
    d = result(name)
    inventory = read(root / "checksums.json")
    panel = read(read(root / "registration.json")["material"])["rows"]
    with (paper / "action-preference-conversations.csv").open("w") as f:
        out = csv.writer(f)
        out.writerow(
            [
                "id",
                "anchor",
                "source",
                "kind",
                "arm",
                "user_messages",
                "answers",
                "callbacks_per_exchange",
                "statuses",
                "asked_for_missing",
                "premature_call",
                "exact_final",
                "grounded_success",
                "seconds",
                "sum_first_token_s",
                "peak_bytes",
            ]
        )
        for side in ("candidate", "control"):
            path = root / (side + ".json.gz")
            assert digest(path) == inventory[path.name]
            for w, r in zip(panel, unpack(path), strict=True):
                assert w["id"] == r["id"]
                xs = r["exchanges"]
                s = r["score"]
                events = [
                    [
                        {"call": t.get("call"), "observation": t.get("observation")}
                        for t in x["turns"]
                        if "call" in t
                    ]
                    for x in xs
                ]
                out.writerow(
                    [
                        r["id"],
                        w.get("anchor"),
                        w.get("domain", "BFCL"),
                        w["kind"],
                        side,
                        compact(w["users"]),
                        compact([x["answer"] for x in xs]),
                        compact(events),
                        compact([x["status"] for x in xs]),
                        s["asked"],
                        s["premature"],
                        s["exact"],
                        s["success"],
                        r["seconds"],
                        r["first"],
                        r["peak"],
                    ]
                )
    from loopqwen.evaluation.relations import cluster_randomization

    document_totals = {}
    for item in panel:
        if item["kind"] != "complete":
            continue
        name = item["call"]["arguments"]["document"]
        j = item["anchor"]
        totals = document_totals.setdefault(name, [0, 0, 0])
        totals[0] += 1
        totals[1] += int(d["pairs"]["candidate"][j])
        totals[2] += int(d["pairs"]["control"][j])
    probability = cluster_randomization([a - b for _, a, b in document_totals.values()])
    with (paper / "action-preference-documents.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "document",
                "anchors",
                "candidate_pairs",
                "base_pairs",
                "all_files_cluster_randomization_p",
            ]
        )
        for document, counts in sorted(document_totals.items()):
            writer.writerow([document, *counts, probability])
    fig, ax = plt.subplots(figsize=(6.7, 2.45))
    labels = [
        "Both file variants",
        "Fully specified",
        "Asked for missing input",
        "Completed after follow-up",
        "Public numerical calls",
    ]
    for side, offset, marker, color, label in (
        ("candidate", -0.10, "o", "#146558", "Preference-trained"),
        ("control", 0.10, "s", "#99552d", "Base"),
    ):
        g = d["groups"]
        nums = [
            sum(d["pairs"][side]),
            g["complete"][side]["success"],
            g["missing"][side]["asked"],
            g["missing"][side]["success"],
            g["public"][side]["success"],
        ]
        ax.scatter(
            [100 * n / z for n, z in zip(nums, (16, 16, 16, 16, 22), strict=True)],
            [i + offset for i in range(5)],
            s=34,
            marker=marker,
            color=color,
            label=label,
        )
    ax.set_yticks(range(5), labels)
    ax.invert_yaxis()
    ax.set_xlim(-4, 104)
    ax.set_xlabel("Share of all enrolled requests or pairs (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=2, frameon=False)
    finish(fig, "action-preference")
