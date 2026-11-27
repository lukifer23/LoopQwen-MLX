"""Register natural source-backed questions, open instructions and prior tool retention together."""

from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen import qwen_tools
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.relational_tools import tools
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.table_worlds import mount
from scripts.grounded_table_sentences import FOLDER as FETA
from scripts.program_language_material import DEST, PANEL, table_prompt
from scripts.table_policy_material import persist

NEW = Path("research/data/conditional-program-panel.json")
PLAN = Path("research/diagnostics/conditional-program-reading.json")
NAME = "conditional-program-primary-01"


def fresh():
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    rows = []

    def add(key, family, source, question, defs, reference, **extra):
        ids = tok.encode(
            qwen_tools.render(tok, [{"role": "user", "content": question}], defs),
            add_special_tokens=False,
        )
        rows.append(
            dict(
                id=key,
                family=family,
                cluster=source,
                source=source,
                questions=[question],
                tools=defs,
                initial_ids=ids,
                reference=reference,
                **extra,
            )
        )

    with TemporaryDirectory() as tmp:
        for r in read(FETA / "originals.json")["test"]:
            key = "feta-" + str(r["id"])
            path = mount(Path(tmp) / key, r["rows"])
            declared = qwen_tools.declarations(tools([path]))
            for mode in ("mounted", "inline"):
                add(
                    key + "-" + mode,
                    "natural_" + mode,
                    key,
                    table_prompt(r, mode == "inline"),
                    declared if mode == "mounted" else [],
                    r["reference"],
                    row_id=r["id"],
                    attested=[a["normalized"] for a in r["attested_values"]],
                )
    for r in read(DEST / "fresh-requests.json"):
        add(
            r["id"],
            "instruction",
            r["id"],
            r["messages"][0]["content"],
            [],
            r["target"]["content"],
            category=r["category"],
        )
    return sorted(rows, key=lambda w: fingerprint([753190, "fresh-order", w["id"]]))


def build():
    addition = fresh()
    existing = read(PANEL)["rows"]
    assert len(addition) == 72 and len(existing) == 268
    # Spread the new sources along the stream. Retain the old requests' internal order.
    rows = []
    k = 0
    for i, w in enumerate(existing, 1):
        rows.append(w)
        while k < len(addition) and (k + 1) * len(existing) <= i * len(addition):
            rows.append(addition[k])
            k += 1
    assert k == 72 and len(rows) == len({w["id"] for w in rows}) == 340
    persist(
        NEW,
        {
            "rows": rows,
            "original_panel_sha256": digest(PANEL),
            "new_source_hashes": {
                str(p): digest(p)
                for p in (
                    DEST / "fresh-requests.json",
                    FETA / "originals.json",
                    FETA / "source-review.json",
                )
            },
        },
    )
    masked = [
        w
        for w in rows
        if w["family"]
        in (
            "natural_mounted",
            "natural_inline",
            "instruction",
            "changed_table",
            "database",
            "open",
        )
    ]
    assert len(masked) == 156
    persist(
        PLAN,
        {
            "ordered_requests": [w["id"] for w in masked],
            "material": digest(NEW),
            "families": dict(Counter(w["family"] for w in masked)),
            "rules": "One project reviewer sees each original request, prompt context or complete table, reference, source qualifications, and visible replies deduplicated and hash shuffled per request. Arm identities, actual tool trajectories, completion times, and all automatic counts remain sealed until every distinct reply has a fully usable, partial, wrong or absent rating, material unsupported flag, request-specific reason and comparative preference. A fully usable natural-table response must answer what was actually asked, accurately convey its key values and relationships, and avoid unsupported additions. Follow the previously sealed 24 source qualifications, including the erroneous year. A well-supported number or pair of cell strings alone does not establish this. For Dolly evaluate fulfillment, accuracy, format and useful detail. References can be subjective, stale or wrong. Accept accurate dated answers and sensible clarification when original time scope is ambiguous. For sources on both views grade each exposed request independently. Grounding on mounted questions also requires a real untruncated read with at least two distinct source-attested values in returned cells. Report this conservative lexical evidence test separately from the meaning of a visible answer. Rate earlier changed-table, database and open requests using their original source-visible protocols. Equal-quality answers may tie. Project authorship and residual stylistic cues limit evaluator independence.",
        },
    )
    print(len(rows), "requests,", len(masked), "masked readings;", digest(NEW))


if __name__ == "__main__":
    build()
