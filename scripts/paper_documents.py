"""Show evidence availability separately from model-initiated search and final answers."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from document_search_material import open_panel

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read
from loopqwen.research.bridge_reading import units
from loopqwen.tool_session import compact


def make(paper, results, result, finish, name="document-search-field-01"):
    spec = read(Path("research/experiments") / (name + ".json"))
    mat = read(spec["material"])
    trained = "initial_adapter" in spec
    prefix = "document-evidence" if trained else "document-search"
    panel = mat["rows"]
    available = {}
    with (
        open_panel(mat) as (search,),
        (
            paper
            / (
                "document-evidence-retrieval.csv"
                if trained
                else "document-retrieval.csv"
            )
        ).open("w") as stream,
    ):
        out = csv.writer(stream)
        out.writerow(
            [
                "id",
                "source",
                "document",
                "question",
                "source_characters",
                "annotated_ranges",
                "ranks_with_reference",
            ]
        )
        lengths = {d["document"]: d["characters"] for d in mat["documents"]}
        for w in panel:
            records = search.callback(query=w["question"], document=w["document"])[
                "passages"
            ]
            ranks = [
                i
                for i, p in enumerate(records, 1)
                if any(p["start"] <= a < b <= p["stop"] for a, b in w["answer_ranges"])
            ]
            available[w["id"]] = bool(ranks)
            out.writerow(
                [
                    w["id"],
                    w["source"],
                    w["document"],
                    w["question"],
                    lengths[w["document"]],
                    compact(w["answer_ranges"]),
                    compact(ranks),
                ]
            )
    root = results / name
    if not (root / "checksums.json").exists():
        return
    decision = result(name)
    sums = read(root / "checksums.json")
    with (paper / (prefix + "-responses.csv")).open("w") as stream:
        out = csv.writer(stream)
        out.writerow(
            [
                "id",
                "source",
                "document",
                "side",
                "question",
                "references",
                "answer",
                "called",
                "evidence",
                "exact",
                "f1",
                "grounded",
                "reference_phrase_in_completed_answer",
                "status",
                "calls",
                "seconds",
                "first_s",
                "peak_bytes",
            ]
        )
        for side in ("candidate", "control"):
            path = root / (side + ".json.gz")
            assert digest(path) == sums[path.name]
            for w, r in zip(panel, unpack(path), strict=True):
                assert w["id"] == r["id"]
                tr = r["trajectory"]
                sc = r["score"]
                words = " " + " ".join(units(tr["answer"] or "")) + " "
                references = [units(ref) for ref in w["refs"]]
                phrase = sc["completed"] and any(
                    ref and " " + " ".join(ref) + " " in words for ref in references
                )
                calls = [
                    {
                        "call": t["call"],
                        "hits": [
                            [h["document"], h["start"], h["stop"]]
                            for h in t["observation"].get("passages", [])
                        ],
                    }
                    for t in tr["turns"]
                    if "call" in t
                ]
                out.writerow(
                    [
                        r["id"],
                        w["source"],
                        w["document"],
                        side,
                        w["question"],
                        compact(w["refs"]),
                        tr["answer"],
                        sc["called"],
                        sc["evidence"],
                        sc["correct"],
                        sc["f1"],
                        sc["grounded"],
                        phrase,
                        tr["status"],
                        compact(calls),
                        r["seconds"],
                        r["first"],
                        r["peak"],
                    ]
                )
    sources = ["SQuAD", "Qasper"] + (["Project guides"] if trained else [])
    fig, axes = plt.subplots(1, len(sources), figsize=(7.15, 2.55), sharey=True)
    for ax, source in zip(axes, sources, strict=True):
        g = decision["groups"][source]
        for i, (key, label) in enumerate(
            (
                ("called", "Tool called"),
                ("evidence", "Reference seen"),
                ("correct", "Exact answer"),
                ("grounded", "Grounded exact"),
            )
        ):
            for s, shift, c, m, title in (
                ("candidate", -0.12, "#11675d", "o", "Recurrent"),
                ("control", 0.12, "#9d5932", "s", "Base"),
            ):
                ax.scatter(
                    g[s][key],
                    i + shift,
                    s=28,
                    color=c,
                    marker=m,
                    label=title if i == 0 else None,
                )
        ax.set_xlim(-0.75, 16.75)
        ax.set_xticks([0, 4, 8, 12, 16])
        available_count = sum(
            available[w["id"]] for w in panel if w["source"] == source
        )
        ax.set_title(
            f"{source} · verbatim query {available_count}/{g['candidate']['n']}",
            fontsize=8,
        )
    axes[0].set_yticks(
        range(4), ["Tool called", "Reference seen", "Exact answer", "Grounded exact"]
    )
    axes[0].invert_yaxis()
    axes[0].legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(-0.03, 1.12),
        ncol=2,
        fontsize=8,
    )
    finish(fig, prefix)
