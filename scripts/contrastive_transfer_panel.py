"""Freeze one direct comparison on held source changes and full, unabridged locale releases."""

from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest
from loopqwen.qwen_tools import declarations, render
from loopqwen.relational_tools import tools
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.table_worlds import canonical, mount, prompt
from scripts.contrastive_world_material import OUT, SEED, ordered
from scripts.locale_context_material import SOURCE as XML
from scripts.locale_context_material import material as locales
from scripts.locale_context_trace import source_request
from scripts.table_policy_material import persist

PANEL = OUT / "transfer-panel.json.gz"
PLAN = Path("research/diagnostics/contrastive-world-transfer.json")


def service_table(w):
    columns = sorted({k for row in w["observation"] for k in row})
    assert columns
    return [
        columns,
        *[
            [str(row[k]) if k in row else None for k in columns]
            for row in w["observation"]
        ],
    ]


def support_prompt(w):
    history = "\n".join(
        ("Customer: " if m["role"] == "user" else "Assistant: ") + m["content"]
        for m in w["messages"]
    )
    return (
        "Continue this customer conversation. The returned "
        + w["service"].split("_")[0].lower()
        + " search results are stored in the attached database. Check the results and reply directly to the customer with a suitable option, including at least two details from the same result.\n\n"
        + history
    )


def build():
    source = unpack(OUT / "withheld.json.gz")
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    rows = []
    with TemporaryDirectory() as tmp:
        db = mount(Path(tmp), [["column"], ["value"]])
        # Declarations depend only on the supplied filename. Each live database below uses the
        # actual held table with its original columns and tuples, independently mounted per task.
        defs = declarations(tools([db]))
        for w in source["tables"]:
            base = w["versions"][0]
            other = min(
                (
                    v
                    for v in w["versions"][1:]
                    if canonical(v["answer"]) != canonical(base["answer"])
                ),
                key=lambda v: fingerprint([SEED, "transfer", w["id"], v["world"]]),
            )
            for v in (base, other):
                rows.append(
                    {
                        "id": w["id"] + "/" + str(v["world"]),
                        "family": "tables",
                        "cluster": w["context"],
                        "questions": [prompt(w)],
                        "source": w["id"],
                        "world": v["world"],
                        "gold": v["answer"],
                        "rows": v["rows"],
                        "defs": defs,
                    }
                )
        for w in source["qualitative"]:
            options = ordered(w["members"][0]["choices"], w["id"])
            for m in w["members"]:
                q = (
                    m["background"]
                    + "\n\n"
                    + m["question"]
                    + "\n"
                    + " or ".join(options)
                    + "?"
                )
                rows.append(
                    {
                        "id": m["id"],
                        "family": "qualitative",
                        "cluster": w["source_background"],
                        "questions": [q],
                        "choices": options,
                        "gold": m["correct"],
                        "defs": [],
                    }
                )
        for pair in source["services"]:
            cluster = fingerprint([w["id"] for w in pair])
            for w in pair:
                rows.append(
                    {
                        "id": w["id"],
                        "family": "service",
                        "cluster": cluster,
                        "questions": [support_prompt(w)],
                        "gold": w["reply"],
                        "rows": service_table(w),
                        "supports": w["supports"],
                        "defs": defs,
                    }
                )
    for w in locales():
        rows.append(
            {
                "id": w["id"],
                "family": "locale",
                "cluster": w["locale"],
                "questions": [source_request(w)],
                "initial_sha256": w["input_sha256"],
                "gold": w["answer"],
                "defs": [],
            }
        )
    for r in rows:
        prompt_text = render(
            tok, [{"role": "user", "content": r["questions"][0]}], r["defs"]
        )
        ids = tok.encode(prompt_text, add_special_tokens=False)
        if r["family"] == "locale":
            assert r["initial_sha256"] == fingerprint(ids) and len(ids) > 131072
        r["initial_ids"] = ids
    assert Counter(r["family"] for r in rows) == {
        "tables": 48,
        "qualitative": 48,
        "service": 48,
        "locale": 4,
    }
    order = sorted(rows, key=lambda r: fingerprint([SEED, "transfer order", r["id"]]))
    manifest = {
        "rows": order,
        "withheld": digest(OUT / "withheld.json.gz"),
        "locale_qualification": digest(XML / "qualification.json"),
    }
    if PANEL.exists():
        assert unpack(PANEL) == manifest
    else:
        packed(PANEL, manifest)
    persist(
        PLAN,
        {
            "material": digest(PANEL),
            "ordered_requests": [r["id"] for r in order],
            "items": len(order),
            "masked_services": [r["id"] for r in order if r["family"] == "service"],
            "review": "Before inspecting model identities or automatic results, review both distinct replies with the full human conversation, all original service results, original human reply and source fields. Grade a useful answer when it responds to the actual request and describes one compatible source result with two accurate details. Accept other valid source options. Record unsupported claims and preferences and one source-specific reason. For grounded credit a real executed database result must contain the supporting values. The published service originally used a simulator. This test mounts its immutable returned records in SQLite and executes real calls on the held data. No prerecorded tool response is substituted for a live invocation.",
            "counts": dict(Counter(r["family"] for r in order)),
        },
    )
    print(len(rows), digest(PANEL), flush=True)


if __name__ == "__main__":
    build()
