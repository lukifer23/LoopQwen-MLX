"""Pair quality and service cost for selected, separately identified systems versus ordinary Base."""

import csv


def write(paper, results, result):
    # Each row names exactly one candidate, denominator and semantic endpoint.
    rows = [
        (
            "tool-dialogue-counterfactual-01",
            "Observation-trained tools",
            "grounded",
            132,
            "candidate",
            "control",
            "success",
        ),
        (
            "conversation-followup-modules-01",
            "Module follow-ups",
            "grounded",
            16,
            "candidate",
            "control",
            "success",
        ),
        (
            "operand-planner-01",
            "Calculator planner",
            "final number",
            48,
            "candidate",
            "control",
            "correct_answer",
        ),
        (
            "bridge-reading-01",
            "First document reader",
            "answer phrase",
            40,
            "candidate",
            "control",
            "content",
        ),
        (
            "linked-reading-01",
            "Linked document reader",
            "answer phrase",
            48,
            "candidate",
            "control",
            "content",
        ),
        (
            "table-parser-01",
            "Table parser",
            "grounded",
            48,
            "candidate",
            "control",
            "success",
        ),
        (
            "tool-dialogue-intent-pairs-01",
            "Resource and tool adapter",
            "grounded",
            78,
            "candidate",
            "control",
            "success",
        ),
        (
            "meaning-alignment-01",
            "Semantic alignment",
            "free judgment",
            80,
            "candidate",
            "control",
            "correct",
        ),
        (
            "document-search-field-01",
            "User-file passage search",
            "grounded exact",
            32,
            "candidate",
            "control",
            "grounded",
        ),
        (
            "report-calculation-01",
            "Stepwise calculator",
            "final number",
            48,
            "candidate",
            "control",
            "correct",
        ),
        (
            "register-mixed-weight-full-01",
            "8-bit feed-forward, 131K",
            "exact",
            2,
            "direct_pages",
            "base",
            "correct",
        ),
        (
            "draft-program-01",
            "Draft then program",
            "final number",
            48,
            "candidate",
            "control",
            "correct_answer",
        ),
        (
            "action-preference-01",
            "Preference-trained tools",
            "grounded",
            54,
            "candidate",
            "control",
            "success",
        ),
        (
            "folded-system-01",
            "Single folded backbone",
            "mixed",
            92,
            "candidate",
            "control",
            "success",
        ),
        (
            "release-mixed-field-01",
            "Released assistant",
            "mixed",
            50,
            "candidate",
            "control",
            "correct",
        ),
        (
            "register-page-full-01",
            "BF16 pages, 131K",
            "exact",
            2,
            "pages",
            "base",
            "correct",
        ),
        (
            "register-packed-direct-full-01",
            "Direct 8-bit pages, 131K",
            "exact",
            2,
            "direct_pages",
            "base",
            "correct",
        ),
        (
            "register-fused-base-full-01",
            "Fused decode, 131K",
            "exact",
            2,
            "direct_pages",
            "base",
            "correct",
        ),
        (
            "narrative-full-book-01",
            "Whole novel, 162K",
            "exact",
            2,
            "candidate",
            "control",
            "correct",
        ),
    ]
    with (paper / "system-comparisons.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "study",
                "system",
                "endpoint",
                "n",
                "candidate",
                "base",
                "seconds_candidate",
                "seconds_base",
                "peak_candidate",
                "peak_base",
                "gate",
            ]
        )
        for study, label, endpoint, n, first, second, key in rows:
            if not (results / study / "checksums.json").is_file():
                continue
            d = result(study)
            if study == "release-mixed-field-01":
                arms = d["by_family"]["all"]
            elif "groups" in d:
                arms = d["groups"]["all"]
            else:
                arms = d["metrics"]
            a, b = (arms[s] for s in (first, second))
            peak = "peak_bytes" if "peak_bytes" in a else "peak"
            w.writerow(
                [
                    study,
                    label,
                    endpoint,
                    n,
                    a[key],
                    b[key],
                    f"{a['seconds']:.2f}",
                    f"{b['seconds']:.2f}",
                    f"{a[peak] / 2**30:.2f}",
                    f"{b[peak] / 2**30:.2f}",
                    "Pass" if d["passed"] else "Miss",
                ]
            )
