"""Exercise every real evaluation callback and scorer with expert and adversarially coincidental returns."""

from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest
from loopqwen.research.provider_field import FieldDatabases
from scripts.provider_field_mutations import LOCAL, OUTPUT, quoted
from scripts.provider_recovery_field import PANEL
from scripts.table_policy_material import persist


def run():
    controls = []
    with TemporaryDirectory() as tmp:
        env = FieldDatabases(Path(tmp))
        for panel in (PANEL, LOCAL):
            rows = unpack(panel)["rows"]
            for r in rows:
                provider = env.providers(r)[1]
                name = r["source"] + ".sqlite"
                call = {
                    "name": provider.name,
                    "arguments": {"database_name": name, "statement": r["expert_sql"]},
                }
                actual = provider.callback(**call["arguments"])
                assert actual == r["expert_result"]
                tr = {
                    "status": "answered",
                    "answer": r["human"],
                    "turns": [{"call": call, "observation": actual}],
                }
                positive = env.score(r, tr)
                direct = env.score(r, {**tr, "turns": []})
                assert positive["success"] and direct["success"] == (
                    r["condition"] == "valid_current"
                )
                report = {"id": r["id"], "expert": positive, "no_new_call": direct}
                if r["condition"] == "unassisted":
                    partner = next(
                        w
                        for w in rows
                        if w["source"] == r["source"]
                        and w["condition"] == "unassisted"
                        and w["dialogue"] != r["dialogue"]
                    )
                    gold = r["gold"]
                    literal = (
                        str(gold)
                        if type(gold) in (float, int)
                        else "'" + str(gold).replace("'", "''") + "'"
                    )
                    constant = (
                        "SELECT "
                        + literal
                        + " FROM "
                        + quoted(actual["referenced_tables"][0])
                        + " LIMIT 1"
                    )
                    assert provider.callback(database_name=name, statement=constant)[
                        "rows"
                    ] == [[gold]]
                    # SQLite values may be recorded as strings despite numeric semantics.
                    for key, query in (
                        ("constant", constant),
                        ("partner", partner["expert_sql"]),
                    ):
                        test = env.program(r, query)
                        assert not test["equivalent_on_suite"]
                        report[key] = test
                controls.append(report)
    out = {
        "panels": {str(p): digest(p) for p in (PANEL, LOCAL, OUTPUT)},
        "rows": controls,
        "expert_complete": sum(r["expert"]["success"] for r in controls),
        "no_new_call_correct": sum(r["no_new_call"]["success"] for r in controls),
        "constant_programs_rejected": sum("constant" in r for r in controls),
        "partner_programs_rejected": sum("partner" in r for r in controls),
    }
    persist(PANEL.with_name("field-score-controls.json"), out)
    print({k: v for k, v in out.items() if k not in ("rows", "panels")})


if __name__ == "__main__":
    run()
