"""Finish the new complete-document candidate and score it against untouched Base."""

import argparse
from pathlib import Path

from loopqwen.io import digest, read, save
from scripts.middle_depth_budget import ROOT
from scripts.middle_depth_decision import contrast, panel, product_gates
from scripts.middle_depth_stages import RUN, documents, final_books, verify_long_fit
from scripts.middle_depth_wave import dispatch, generate

SCOPE = Path("research/diagnostics/middle-depth/direct-pair.md")
CHOSEN = ROOT / "direct-decision.json"
SHORT = "direct-full-fit"
LONG = "direct-child"


def parent():
    r = read(ROOT / "history-decision.json")
    view = r["models"]["loop"]
    path = Path(view["path"])
    if digest(path / "adapters.safetensors") != view["weights"]:
        raise ValueError("The short-trained loop weights changed")
    return r["stage"], path


def children():
    _, src = parent()
    child = RUN / "long/loop"
    proof = verify_long_fit("loop", src, child)
    save(ROOT / "direct-long-integrity.json", proof)
    from scripts.middle_depth_standalone import full_fit_bundle

    bundle = full_fit_bundle(child)
    generate(SHORT, ["loop"], [128, 512], {"loop": bundle})
    documents(LONG, {"loop": bundle}, ["loop"], [128, 512])


def select():
    previous, src = parent()
    base = panel("base", "terminal", "parent", 512)
    scores = {}
    for key, short, long in (("parent", previous, "parent"), ("child", SHORT, LONG)):
        r = contrast(panel("loop", short, long, 512), base)
        scores[key] = {
            "short": {
                f: r["families"][f]["left"] for f in ("chat", "tools", "reasoning")
            },
            "documents": {s: v["left"] for s, v in r["book_bands"].items()},
        }
    a, b = scores["child"], scores["parent"]
    keep = (
        sum(a["documents"].values()) > sum(b["documents"].values())
        and all(a["documents"][s] >= v for s, v in b["documents"].items())
        and sum(a["short"].values()) >= sum(b["short"].values())
    )
    model = RUN / "long/loop" if keep else src
    save(
        CHOSEN,
        {
            "scope": digest(SCOPE),
            "retained_full_fit": keep,
            "evidence": scores,
            "short_stage": SHORT if keep else previous,
            "document_stage": LONG if keep else "parent",
            "models": {
                "loop": {
                    "path": str(model),
                    "weights": digest(model / "adapters.safetensors"),
                }
            },
        },
    )
    decisions()


def decisions(parent_only=False):
    d = (
        {"scope": digest(SCOPE), "short_stage": parent()[0], "document_stage": "parent"}
        if parent_only
        else read(CHOSEN)
    )
    label = "parent" if parent_only else "chosen"
    if d["scope"] != digest(SCOPE):
        raise ValueError("The selected research question changed")
    for cap in (128, 512):
        candidate = panel("loop", d["short_stage"], d["document_stage"], cap)
        base = panel("base", "terminal", "parent", cap)
        result = contrast(candidate, base)
        save(
            ROOT / "direct" / f"{label}-{cap}.json",
            {
                "scope": d["scope"],
                "models": ["loop", "base"],
                "cap": cap,
                "product": result,
                "development_thresholds": product_gates(result),
                "two_point_development_lower_bound_passes": result["paired_interval"][0]
                > -0.02,
            },
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "stage",
        choices=(
            "fit",
            "children",
            "select",
            "primary-books",
            "final-books",
            "decisions",
            "parent-scores",
        ),
    )
    stage = ap.parse_args().stage
    if stage == "fit":
        dispatch("long_fit", "fit", "--arm", "loop", "--policy", parent()[1])
    elif stage == "children":
        children()
    elif stage == "select":
        select()
    elif stage in ("primary-books", "final-books"):
        final_books(
            ("loop", "base"),
            CHOSEN.name,
            "direct-complete-books.json",
            primary_only=stage == "primary-books",
        )
    else:
        decisions(stage == "parent-scores")
