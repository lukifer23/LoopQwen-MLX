"""Physically replay original calculator annotations and complete locale evidence before inference."""

from loopqwen.calculator_tool import calculate, tool
from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, save
from loopqwen.qwen_tools import declarations, render
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.shared_query_transfer import score
from scripts.locale_context_material import material
from scripts.locale_context_trace import source_request
from scripts.register_shared_query import PANEL
from scripts.shared_query_material import ROOT


def build():
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    base = {w["id"]: w for w in unpack(ROOT / "held.json.gz")}
    long = {w["id"]: w for w in material()}
    results = []
    reports = []
    strategy = []
    for w in unpack(PANEL)["rows"]:
        key = w["id"]
        f = w["family"]
        defs = declarations([tool()]) if f == "report" else []
        assert w["defs"] == defs
        question = w["questions"][0]
        p = tok.encode(
            render(tok, [{"role": "user", "content": question}], defs),
            add_special_tokens=False,
        )
        assert p == w["initial_ids"] and fingerprint(p) == w["initial_sha256"]
        if f == "locale":
            r = long[key]
            assert (
                question == source_request(r)
                and w["gold"] == r["answer"]
                and min(r["pattern_start_tokens"]) >= 131072
            )
            results.append(
                {
                    "id": key,
                    "tokens": len(p),
                    "evidence_positions": r["pattern_start_tokens"],
                    "source": r["file_sha256"],
                    "prompt_sha256": fingerprint(p),
                }
            )
        elif f == "report":
            orig = base[key]
            assert question == orig["request"]
            obs = calculate(w["steps"])
            assert obs == orig["observed"]
            event = {
                "call": {"name": "calculate", "arguments": {"steps": w["steps"]}},
                "observation": obs,
            }
            rated = score(
                w, {"status": "answered", "answer": w["reference"], "turns": [event]}
            )
            assert (
                rated["grounded"]
                and rated["expert_values"]
                and rated["arithmetic_path"]
            ), key
            bad = score(
                w, {"status": "answered", "answer": w["reference"], "turns": []}
            )
            assert not bad["success"] and not bad["arithmetic_path"]
            reports.append(
                {
                    "id": key,
                    "issuer": w["cluster"],
                    "native_input": len(p),
                    "executed": fingerprint(obs),
                }
            )
        else:
            orig = base[key]
            assert question == orig["request"]
            yes = score(
                w, {"status": "answered", "answer": orig["reference"], "turns": []}
            )
            no = score(
                w,
                {
                    "status": "answered",
                    "answer": "No." if orig["label"] else "Yes.",
                    "turns": [],
                },
            )
            assert yes["success"] and not no["success"]
            strategy.append(
                {
                    "id": key,
                    "topic": w["cluster"],
                    "native_input": len(p),
                    "label": w["label"],
                }
            )
    assert (
        len({x["issuer"] for x in reports}) == 32
        and len({x["topic"] for x in strategy}) == 48
    )
    assert sum(x["label"] for x in strategy) == 24 and len(results) == 4
    save(
        ROOT / "panel-replay.json",
        {
            "panel_sha256": digest(PANEL),
            "reports": reports,
            "strategy": strategy,
            "locale": results,
        },
    )
    print(
        "Verified",
        len(reports),
        "expert calculations,",
        len(strategy),
        "human conclusions and",
        len(results),
        "complete distant-source inputs.",
    )


if __name__ == "__main__":
    build()
