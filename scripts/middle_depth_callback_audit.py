"""Independently replay every recorded short-panel tool observation and check credit."""

from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from scripts.middle_depth_budget import ROOT
from scripts.middle_depth_development import SOURCE, instance


def audit(stages=("terminal", "quantized"), arms=None, destination=None):
    cases = {r["id"]: r for r in read(SOURCE) if r.get("provider")}
    # Reference quantities from the enrolled invoices and CSV, independent of the model.
    targets = {
        "tiles": ("184", "174", "175"),
        "ink": ("190", "270", None),
        "parcels": ("Loft", "Annex", "Friday"),
    }
    output = []
    with TemporaryDirectory() as t:
        for stage in stages:
            for row in read(ROOT / "review" / stage / "scored.json"):
                source = row["source"]
                world = row["world"]
                if (
                    (arms is not None and row["arm"] not in arms)
                    or row["cap"] != 512
                    or source not in cases
                ):
                    continue
                path = ROOT / row["file"]
                assert digest(path) == row["sha"]
                _, providers = instance(
                    cases[source],
                    world
                    if world in ("original", "changed", "irrelevant")
                    else "original",
                    Path(t),
                )
                lookup = {p.name: p for p in providers}
                target = targets[source][
                    2 if world in ("own", "shared") else int(world == "changed")
                ]
                actual = []
                submitted = []
                for turn in unpack(path)["trace"]["turns"]:
                    calls = [turn["call"]] if "call" in turn else turn.get("calls", [])
                    obs = (
                        [turn["observation"]]
                        if "observation" in turn
                        else turn.get("observations", [])
                    )
                    submitted.extend(calls)
                    if not obs:
                        continue
                    for call, witness in zip(calls, obs, strict=True):
                        try:
                            replay = lookup[call["name"]].callback(**call["arguments"])
                        except (
                            ValueError,
                            SyntaxError,
                            ArithmeticError,
                            TypeError,
                        ) as ex:
                            replay = {"error": type(ex).__name__}
                        if replay != witness:
                            raise ValueError(f"Live provider differs from {path}")
                        actual.append(replay)
                available = any(str(x.get("value")) == target for x in actual)
                if row["grade"]["correct"] and not available:
                    raise ValueError(f"Unsupported credit on {path}")
                output.append(
                    {
                        "stage": stage,
                        "arm": row["arm"],
                        "source": source,
                        "world": world,
                        "status": row["status"],
                        "target": target,
                        "submitted": submitted,
                        "replayed": actual,
                        "target_observed": available,
                        "correct": row["grade"]["correct"],
                        "file": row["file"],
                        "sha": row["sha"],
                    }
                )
    save(
        destination or ROOT / "quantization" / "callback-audit.json",
        {
            "rows": output,
            "presentations": len(output),
            "observations": sum(len(r["replayed"]) for r in output),
            "all_provider_results_reproduced": True,
            "graded_success_without_target_observation": 0,
        },
    )
    for stage, arm in sorted({(r["stage"], r["arm"]) for r in output}):
        rows = [
            r
            for r in output
            if (r["stage"], r["arm"]) == (stage, arm)
            and r["world"] not in ("own", "shared")
        ]
        print(
            stage,
            arm,
            len(rows),
            sum(r["target_observed"] for r in rows),
            sum(r["correct"] for r in rows),
            dict(Counter(r["status"] for r in rows)),
        )


def first_calculations():
    """Check the first emitted arithmetic action before any provider feedback."""
    from loopqwen.calculator_tool import tool

    reference = ROOT / "direct/callback-replay.json"
    providers = {
        "registered": tool(),
        "checked_equalities": tool(check_equalities=True),
    }
    rows = []
    for row in read(reference)["rows"]:
        if row["source"] not in ("ink", "tiles") or row["world"] in ("own", "shared"):
            continue
        first = next(iter(row["submitted"]), None)
        outcomes = {}
        for mode, provider in providers.items():
            if first is None or first["name"] != provider.name:
                outcomes[mode] = {"status": "no calculator action"}
                continue
            try:
                result = provider.callback(**first["arguments"])
            except (ValueError, TypeError, SyntaxError, ArithmeticError) as e:
                result = {"error": type(e).__name__}
            if mode == "registered" and result != row["replayed"][0]:
                raise ValueError(f"First observation changed for {row['file']}")
            status = (
                "rejected"
                if "error" in result
                else (
                    "returned final target"
                    if result["value"] == row["target"]
                    else "returned other quantity"
                )
            )
            outcomes[mode] = {"status": status, "result": result}
        rows.append(
            {
                k: row[k]
                for k in ("stage", "arm", "source", "world", "target", "file", "sha")
            }
            | {"first_action": first, "outcomes": outcomes}
        )
    save(
        ROOT / "direct/first-calculations.json",
        {
            "callback_replay": digest(reference),
            "method": "All six numeric openings per evaluated checkpoint. Execute only the first submitted action with each existing provider, leaving its arguments unchanged. Equality-mode results measure syntax compatibility only. The original live trajectory and primary grade remain the reference.",
            "rows": rows,
        },
    )


def training_openings():
    import ast

    from loopqwen.calculator_tool import tool
    from loopqwen.identity import fingerprint
    from loopqwen.qwen_tools import parse
    from loopqwen.research.cross_view_study import tokenizer
    from scripts.middle_depth_budget import BASE
    from scripts.middle_depth_material import ROWS

    tok = tokenizer({"checkpoint": str(BASE)})
    provider = tool()
    rows, calls_by_history = [], {"opening": Counter(), "after_provider": Counter()}
    for row in unpack(ROWS):
        if row["family"] != "tools":
            continue
        prompt, target = tok.decode(row["p"]), tok.decode(row["y"])
        history = "after_provider" if "<tool_response>" in prompt else "opening"
        visible = target.split("</think>", 1)[1].removesuffix("<|im_end|>").strip()
        calls = parse(
            visible, {"calculate": provider.definition(), "database_query": {}}
        )
        calls_by_history[history].update(a["name"] for a in calls)
        if history != "opening":
            continue
        (call,) = calls
        if call["name"] != provider.name:
            raise ValueError(
                "Expected calculator-only openings in this frozen training set"
            )
        steps = call["arguments"]["steps"]
        products = [
            ast.unparse(node)
            for expression in steps
            for node in ast.walk(ast.parse(expression.strip(), mode="eval"))
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)
        ]
        rows.append(
            {
                "id": row["id"],
                "cluster": row["cluster"],
                "prompt_ids": fingerprint(row["p"]),
                "target_ids": fingerprint(row["y"]),
                "steps": steps,
                "value": provider.callback(**call["arguments"])["value"],
                "multiplications": products,
                "contrasts": [
                    {
                        "position": pos,
                        "target_token": tok.decode([row["y"][pos]]),
                        "foil_token": tok.decode([foil]),
                    }
                    for pos, foil in row["negatives"]
                ],
            }
        )
    save(
        ROOT / "direct/training-first-actions.json",
        {
            "data": digest(ROWS),
            "scope": "All 72 tool targets in the 288-row source-supervised training mixture. Parse the actual native targets, execute each initial calculator call and inspect every multiplication in its expression tree. Provider-history groups follow actual message markup.",
            "calls_by_history": calls_by_history,
            "opening_rows": len(rows),
            "distinct_opening_clusters": len({r["cluster"] for r in rows}),
            "source_corpora": dict(Counter(r["id"].split("/")[0] for r in rows)),
            "opening_multiplications": dict(
                Counter(p for r in rows for p in r["multiplications"])
            ),
            "rows": rows,
        },
    )


def audit_thinking():
    import re
    from statistics import median

    from loopqwen.research.cross_view_study import tokenizer
    from scripts.middle_depth_budget import BASE
    from scripts.middle_depth_material import ROWS

    tok = tokenizer({"checkpoint": str(BASE)})
    details = []
    for r in unpack(ROWS):
        left, close, right = tok.decode(r["y"]).partition("</think>")
        prompt = tok.decode(r["p"])
        details.append(
            {
                "id": r["id"],
                "family": r["family"],
                "group": r["group"],
                "open_thinking": prompt.endswith("<think>\n"),
                "closing": bool(close),
                "thought_text": bool(left.strip()),
                "prior_provider_response": "<tool_response>" in prompt,
                "action_targets": re.findall(r"<function=([^>]+)>", right),
                "target_positions": len(r["y"]),
            }
        )
    families = {}
    for f in ("chat", "reasoning", "documents", "tools"):
        z = [r for r in details if r["family"] == f]
        families[f] = {
            "rows": len(z),
            "open_thinking": sum(r["open_thinking"] for r in z),
            "closing": sum(r["closing"] for r in z),
            "nonempty_thought_targets": sum(r["thought_text"] for r in z),
            "prior_provider_response": sum(r["prior_provider_response"] for r in z),
            "target_calls": dict(Counter(k for r in z for k in r["action_targets"])),
            "target_positions": sum(r["target_positions"] for r in z),
        }
    save(
        ROOT / "training-format-audit.json",
        {
            "weights_experiment": "source-supervised",
            "training": digest(ROWS),
            "definition": "Decode each stored target with the deployed tokenizer. Count non-whitespace text preceding the first closing think marker, and actual native function names in the subsequent target. Distinguish requests that already contain provider history.",
            "families": families,
            "rows": details,
        },
    )
    close = tok.convert_tokens_to_ids("</think>")
    scored = ROOT / "review/terminal/scored.json"
    raw = []
    for m in read(scored):
        if m["cap"] != 512 or m["world"] in ("own", "shared"):
            continue
        path = ROOT / m["file"]
        assert digest(path) == m["sha"]
        t = unpack(path)["trace"]["turns"][0]["token_ids"]
        n = t.index(close) if close in t else len(t)
        raw.append(
            {
                "arm": m["arm"],
                "family": m["family"],
                "source": m["source"],
                "world": m["world"],
                "thought_positions": n,
                "closed": close in t,
                "empty": close in t and not tok.decode(t[:n]).strip(),
                "grade": m["grade"]["correct"],
                "trace": m["sha"],
            }
        )
    by = {}
    for arm in ("loop", "ordinary", "base", "instruction"):
        by[arm] = {}
        for f in ("tools", "reasoning", "chat"):
            z = [r for r in raw if r["arm"] == arm and r["family"] == f]
            by[arm][f] = {
                "n": len(z),
                "empty": sum(r["empty"] for r in z),
                "unclosed": sum(not r["closed"] for r in z),
                "median_thought_positions": median(r["thought_positions"] for r in z),
            }
    save(
        ROOT / "generated-thought-audit.json",
        {
            "scope": "First actual assistant generation on all 27 terminal short openings per arm with 512-token allowance. Counts observed positions up to the closing think token or exhaustion.",
            "cohort": digest(scored),
            "arms": by,
            "rows": raw,
        },
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--direct", action="store_true")
    p.add_argument("--training-openings", action="store_true")
    args = p.parse_args()
    if args.training_openings:
        training_openings()
    elif args.direct:
        audit(
            ("terminal", "direct-full-fit"),
            {"loop", "base"},
            ROOT / "direct/callback-replay.json",
        )
        first_calculations()
    else:
        audit()
        audit_thinking()
