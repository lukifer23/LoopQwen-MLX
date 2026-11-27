"""Rebuild every frozen prompt, execute supplied tool histories, and check no answer markup entered the prompt."""

import json
import sqlite3
from collections import Counter
from tempfile import TemporaryDirectory

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest
from loopqwen.qwen_tools import declarations, render
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.provider_field import FieldDatabases
from scripts.register_repair_query import PANEL, ROOT
from scripts.register_shared_query import PANEL as EARLIER
from scripts.table_policy_material import persist


def run():
    panel = unpack(PANEL)
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    originals = {r["id"]: r for r in unpack(EARLIER)["rows"]}
    c = Counter()
    checks = []
    actual = {}
    with TemporaryDirectory() as tmp:
        fields = FieldDatabases(tmp)
        for w in panel["rows"]:
            question = w["questions"][0]
            if w["family"] != "field":
                assert w == originals[w["id"]]
                messages = [{"role": "user", "content": question}]
            else:
                prov = fields.providers(w)
                assert declarations(prov) == w["defs"]
                catalog = {t.name: t for t in prov}
                messages = w["messages"]
                assert messages[0] == {"role": "user", "content": question}
                if w["condition"] == "unassisted":
                    assert len(messages) == 1
                else:
                    assert len(messages) == 5
                    for idx in (1, 3):
                        (call,) = messages[idx]["tool_calls"]
                        fn = call["function"]
                        try:
                            result = catalog[fn["name"]].callback(**fn["arguments"])
                        except sqlite3.OperationalError:
                            result = {"error": "OperationalError"}
                        assert (
                            messages[idx + 1]["role"] == "tool"
                            and json.loads(messages[idx + 1]["content"]) == result
                        )
                        c["verified_initial_callbacks"] += 1
                    if w["condition"] == "provider_error":
                        assert "error" in result
                    elif w["condition"] == "valid_current":
                        assert result == w["expert_result"]
                    else:
                        assert "error" not in result and result["rows"] != [[w["gold"]]]
                    if w["condition"] in ("valid_current", "valid_other"):
                        key = fingerprint(messages[1:])
                        actual.setdefault((w["source"], key), []).append(
                            (w["question"], w["condition"])
                        )
            ids = tok.encode(render(tok, messages, w["defs"]), add_special_tokens=False)
            assert ids == w["initial_ids"] and fingerprint(ids) == w["initial_sha256"]
            if w["family"] == "locale":
                assert len(ids) > 131072
            c[w["family"]] += 1
            checks.append(
                {"id": w["id"], "opening_tokens": len(ids), "sha256": fingerprint(ids)}
            )
    assert len(actual) == 24 and all(
        len(v) == 2
        and v[0][0] != v[1][0]
        and {x[1] for x in v} == {"valid_current", "valid_other"}
        for v in actual.values()
    )
    assert c == {
        "field": 96,
        "locale": 4,
        "report": 32,
        "strategy": 48,
        "verified_initial_callbacks": 144,
    }
    assert set(originals) <= {r["id"] for r in checks}
    persist(
        ROOT / "panel-replay.json",
        {
            "panel": digest(PANEL),
            "verified": checks,
            "counts": dict(c),
            "identical_histories_with_reversed_correct_action": len(actual),
        },
    )
    print(
        dict(c),
        "long lengths",
        [r["opening_tokens"] for r in checks if r["opening_tokens"] > 131072],
    )


if __name__ == "__main__":
    run()
