"""Check paired conversational inputs and the scoring path without running either model."""

from fractions import Fraction
from pathlib import Path
from loopqwen.calculator_tool import calculate, tool
from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import read, digest
from loopqwen.qwen_tools import render, declarations
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.source_copy_transfer import SCALES, SECOND_AMOUNTS, score
from scripts.register_source_copy import PROTOCOL, common
from scripts.source_copy_material import PANEL, TOKENS
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

OUT = Path("docs/measurements/source-copy-panel.json")


def run():
    for p in (PANEL, TOKENS, PROTOCOL):
        committed(p)
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    qs = unpack(PANEL)
    checked = []
    for w, r in zip(qs, read(PROTOCOL)["order"], strict=True):
        assert w["id"] == r["id"]
        sys = dict(role="system", content=w["system"])
        defs = declarations([tool()]) if w["family"] == "report" else []
        first = [
            sys,
            dict(role="user", content=w["body"] + "\n\n" + w["turns"][0]["question"]),
        ]
        h, q = common(w)
        for msgs, k in (
            (first, "first"),
            ([*h, dict(role="user", content=q)], "fixed"),
        ):
            ids = tok.encode(render(tok, msgs, defs), add_special_tokens=False)
            assert fingerprint(ids) == r[k]
        for i, t in enumerate(w["turns"]):
            if w["family"] == "report":
                steps = t["steps"]
                obs = calculate(steps)
                assert obs == t["observed"]
                val = Fraction(obs["value"])
                scale = (
                    SCALES[w["cluster"]]
                    if i == 0 or w["cluster"] in SECOND_AMOUNTS
                    else None
                )
                suffix = {
                    None: "",
                    0: "",
                    3: " thousand",
                    6: " million",
                    9: " billion",
                }[scale]
                answer = f"{float(val):.5f}" + suffix
                tr = dict(
                    status="answered",
                    answer=answer,
                    turns=[
                        dict(
                            call=dict(name="calculate", arguments=dict(steps=steps)),
                            observation=obs,
                        )
                    ],
                )
                assert score(w, i, tr)["success"], (w["id"], i, answer)
                assert not score(w, i, {**tr, "turns": []})["success"]
                assert not score(
                    w, i, {**tr, "answer": f"{abs(float(val)) + 22222:.5f}" + suffix}
                )["success"]
                if scale is None:
                    assert score(w, i, {**tr, "answer": f"{100 * float(val):.2f}%"})[
                        "success"
                    ]
            else:
                a = t["answers"][0]
                assert score(w, i, dict(status="answered", answer=a, turns=[]))[
                    "success"
                ]
                assert not score(
                    w, i, dict(status="answered", answer="unrelated.", turns=[])
                )["success"]
            checked.append([w["id"], i])
    assert len(checked) == 96
    # Previous study sources use literal PDF page IDs. Scrutinize every earlier
    # registered source/data artifact (case-insensitive), including compressed files.
    import gzip

    panel_pages = {w["file"].casefold().encode() for w in qs if w["family"] == "report"}
    examined = 0
    hit = {}
    for p in sorted(
        set(Path("research/sources").rglob("*.json*"))
        | set(Path("research/data").glob("*.json*"))
    ):
        if "source-copy" in p.parts or not p.is_file():
            continue
        b = gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
        matches = [s.decode() for s in panel_pages if s in b.lower()]
        if matches:
            hit[str(p)] = matches
        examined += 1
    assert not hit, hit
    persist(
        OUT,
        dict(
            passed=True,
            plan=digest(PROTOCOL),
            panel=digest(PANEL),
            items=len(checked),
            matched_openings=len(checked),
            source_files_scanned=examined,
            earlier_literal_page_hits=hit,
            training=digest(TOKENS),
            scorer=digest(Path("loopqwen/research/source_copy_transfer.py")),
            description="96 exact openings and 96 reference answers. All 48 report programs were executed. Both a valid program with the wrong delivered quantity and the correct quantity without a call failed each report test. Complete normalized reference text was required for reading. 24 independent report issuers; 22 were represented by other pages in earlier project panels.",
        ),
    )


if __name__ == "__main__":
    run()
