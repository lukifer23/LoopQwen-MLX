"""Direct, timed evaluation of a single tied-query model and the untouched ordinary Base."""

import gc
import re
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import mlx.core as mx

from ..calculator_tool import tool
from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import digest, read, save
from ..shared_query_assistant import SharedQueryAssistant
from ..tool_compare import DenseAssistant
from .conditional_primary import ARMS, costs
from .contrastive_transfer import score as locale_score
from .core import Budget
from .cross_view_study import tokenizer
from .file_evidence_environment import capture
from .linked_rollin import exact_sign_flip
from .native_function_study import executions, replay
from .provider_field import FieldDatabases
from .registered import registered
from .report_calculation import grade, operation_evidence
from .shared_query_features import FIT, PRIMARY, RESULTS, SOURCES
from .table_reentry_fit import root_for


def score(w, tr):
    completed = tr["status"] == "answered"
    text = tr["answer"] or ""
    if w["family"] == "locale":
        return locale_score(w, tr)
    if w["family"] == "strategy":
        # This is the stated conclusion. The full explanation is assessed under the separate blind gate.
        label = re.match(
            r"^\s*(?:(?:the\s+)?answer\s*(?:is\s+|:\s*))?[\s*_]*(yes|no)\b",
            text,
            flags=re.IGNORECASE,
        )
        picked = label.group(1).lower() == "yes" if label else None
        return {
            "completed": completed,
            "success": completed and picked == w["label"],
            "stated_choice": picked,
        }
    calls = executions(tr)
    simple = {**tr, "turns": calls}
    # Calculation equivalence does not require exactly the expert's expression.
    result = grade({k: v for k, v in w.items() if k != "steps"}, simple)
    valid = [c for c in calls if "value" in c["observation"]]
    arithmetic = False
    for c in valid:
        steps = c["call"]["arguments"]["steps"]
        arithmetic |= any(
            operation_evidence(expr, steps[:i], set())[0]
            for i, expr in enumerate(steps)
        )
    result["arithmetic_path"] = bool(
        valid
        and len(valid) == len(calls)
        and arithmetic
        and abs(Fraction(valid[-1]["observation"]["value"]) - Fraction(w["gold"]))
        <= Fraction(w["tolerance"])
    )
    actual = [Fraction(v) for c in calls for v in c["observation"].get("results", [])]
    expected = [Fraction(v) for v in w["observed"]["results"]]
    result.update(
        expert_values=bool(result["grounded"] and actual == expected),
        success=result["grounded"],
    )
    return result


def providers(w, fields):
    if w["family"] == "field":
        return fields.providers(w)
    return [tool()] if w["family"] == "report" else []


def measure(s, w, arm, actor, guard, fields=None):
    prov = providers(w, fields)
    mx.synchronize()
    mx.reset_peak_memory()
    start = perf_counter()
    if actor is None:
        if arm == "candidate":
            actor = SharedQueryAssistant(
                s["checkpoint"], RESULTS / s.get("fit_stage", FIT)
            )
        else:
            actor = DenseAssistant(True, "qwen", s["checkpoint"])
            assert actor.language.base_source == "certified_release_text"
            actor.native_plain = True
    net = actor._tool_model
    count0 = (net.output_positions, net.output_revisits) if arm == "candidate" else None
    first = None

    def progress(count):
        nonlocal first
        if first is None and count == len(w["initial_ids"]):
            first = perf_counter() - start
        guard.check()

    options = {
        "max_tokens": s["max_tokens"],
        "max_turns": s["context_turns"] if w["family"] == "locale" else s["max_turns"],
        "progress": progress,
        "prefill_cache_state": True,
        "reuse_prefix": True,
        "record_kv_history": True,
    }
    if w["family"] == "field" and w["condition"] != "unassisted":
        tr = actor.continue_after_tool(w["messages"], prov, **options)
    else:
        tr = actor.conversation(prov).respond(w["questions"][0], **options)
    mx.synchronize()
    elapsed = perf_counter() - start
    assert first is not None and tr["turns"][0]["prompt_ids"] == w["initial_ids"]
    q = {
        "trajectory": tr,
        "score": fields.score(w, tr) if w["family"] == "field" else score(w, tr),
        "first_s": first,
        "wall_s": elapsed,
        "peak": mx.get_peak_memory(),
    }
    if count0:
        q["output_positions"] = net.output_positions - count0[0]
        q["read_only_visits"] = net.output_revisits - count0[1]
        assert q["output_positions"] == sum(len(t["token_ids"]) for t in tr["turns"])
        assert q["read_only_visits"] == 2 * q["output_positions"]
    return actor, {"id": w["id"], "family": w["family"], "requests": [q]}


def outcome(rows, runs):
    counts = {}
    c = {arm: costs([r["requests"][0] for r in runs[arm]]) for arm in ARMS}
    gates = {}
    for fam in dict.fromkeys(r["family"] for r in rows):
        pairs = []
        schemas = defaultdict(list)
        conditions = defaultdict(list)
        complete = [0, 0]
        for i, w in enumerate(rows):
            if w["family"] != fam:
                continue
            values = []
            for a, arm in enumerate(ARMS):
                r = runs[arm][i]
                assert w["id"] == r["id"]
                sc = r["requests"][0]["score"]
                values.append(int(sc["success"]))
                complete[a] += sc["completed"]
            pairs.append(values)
            if fam == "field":
                schemas[w["source"]].append(values[0] - values[1])
                conditions[w["condition"]].append(values[0] - values[1])
        changes = [a - b for a, b in pairs]
        totals = [sum(r[j] for r in pairs) for j in (0, 1)]
        # Each short source occurs once. Long pairs use the same original locale across releases.
        p = exact_sign_flip(changes) if fam != "locale" else None
        if schemas:
            p = exact_sign_flip([sum(v) for v in schemas.values()])
        counts[fam] = {
            "total": totals,
            "completed": complete,
            "gains": changes.count(1),
            "losses": changes.count(-1),
            "p": p,
            "n": len(pairs),
        }
        gates[fam] = (
            totals[0] == 4 and totals[0] > totals[1]
            if fam == "locale"
            else sum(changes) >= 6 and -1 not in changes and p <= 0.05
        )
        if schemas:
            gains = {k: sum(v) for k, v in conditions.items()}
            counts[fam]["condition_net"] = gains
            counts[fam]["schema_net"] = {k: sum(v) for k, v in schemas.items()}
            gates["all_field_conditions"] = all(v > 0 for v in gains.values())
        gates[fam + "_completion"] = complete[0] >= complete[1]
    gates.update(
        {k: c["candidate"][k] < c["control"][k] for k in ("peak", "wall_s", "first_s")}
    )
    gates.update(
        {k: c["candidate"][k] > c["control"][k] for k in ("opening_tps", "decode_tps")}
    )
    return {"passed": all(gates.values()), "gates": gates, "counts": counts, "costs": c}


def verify(root, d):
    s = registered(read(root / "registration.json")["id"])
    assert s == read(root / "registration.json")
    if d.get("status") == "interrupted":
        assert (
            not d["passed"]
            and d["reason"]
            and all(
                len(unpack(root / (a + "-partial.json.gz"))) == n
                for a, n in d["durable"].items()
            )
        )
        return
    rows = unpack(Path(s["material"]))["rows"]
    tok = tokenizer(s)
    runs = {a: unpack(root / (a + ".json.gz")) for a in ARMS}
    fit = RESULTS / s.get("fit_stage", FIT)
    assert read(root / "weights.json") == {
        "step": digest(fit / "step.safetensors"),
        "archive": digest(fit / "checksums.json"),
        "backbone": read(fit / "fit.json")["backbone"]["sha256"],
        "base": s["base_text_sha256"],
    }
    with TemporaryDirectory() as tmp:
        fields = FieldDatabases(tmp) if s.get("field_history") else None
        for arm in ARMS:
            for w, r in zip(rows, runs[arm], strict=True):
                (q,) = r["requests"]
                t = q["trajectory"]
                assert (
                    t["turns"][0]["prompt_ids"] == w["initial_ids"]
                    and fingerprint(w["initial_ids"]) == w["initial_sha256"]
                )
                # Both architectures own exactly the six ordinary temporal KV slots.
                replay(
                    tok,
                    w,
                    providers(w, fields),
                    deepcopy(w["messages"])
                    if w["family"] == "field"
                    else [{"role": "user", "content": w["questions"][0]}],
                    q,
                    {
                        **s,
                        "max_turns": s["context_turns"]
                        if w["family"] == "locale"
                        else s["max_turns"],
                    },
                    "shared_memory" if arm == "candidate" else "control",
                )
                if arm == "candidate":
                    assert (
                        q["read_only_visits"]
                        == 2 * q["output_positions"]
                        == 2 * sum(len(z["token_ids"]) for z in t["turns"])
                    )
                assert (
                    fields.score(w, t) if w["family"] == "field" else score(w, t)
                ) == q["score"]
    assert outcome(rows, runs) == d


def run(name=PRIMARY):
    s = registered(name)
    fit = RESULTS / s.get("fit_stage", FIT)
    if not read(fit / "decision.json")["passed"]:
        raise ValueError("The tied-query fit missed its registered prerequisite")
    if s.get("field_history"):
        cold = read(Path("docs/measurements") / (s["fit_stage"] + "-cold.json"))
        assert cold["fit"] == digest(fit / "checksums.json")
    root = root_for(s)
    capture(
        root,
        (
            *SOURCES,
            "loopqwen/research/shared_query_fit.py",
            "loopqwen/research/shared_query_transfer.py",
            "loopqwen/research/native_function_study.py",
            "loopqwen/evaluation/generation_eval.py",
            *(
                (
                    "loopqwen/research/provider_field.py",
                    "loopqwen/tool_session.py",
                    "loopqwen/tool_model.py",
                    "scripts/provider_field_mutations.py",
                )
                if s.get("field_history")
                else ()
            ),
        ),
    )
    save(
        root / "weights.json",
        {
            "step": digest(fit / "step.safetensors"),
            "archive": digest(fit / "checksums.json"),
            "backbone": read(fit / "fit.json")["backbone"]["sha256"],
            "base": s["base_text_sha256"],
        },
    )
    guard = Budget(name, s["limits"])
    runs = {}
    try:
        rows = unpack(Path(s["material"]))["rows"]
        with TemporaryDirectory() as tmp:
            fields = FieldDatabases(tmp) if s.get("field_history") else None
            for arm in ARMS:
                seq = runs[arm] = []
                actor = None
                gc.collect()
                mx.clear_cache()
                for i, w in enumerate(rows, 1):
                    actor, r = measure(s, w, arm, actor, guard, fields)
                    seq.append(r)
                    if i % 8 == 0:
                        packed(root / (arm + "-partial.json.gz"), seq)
                        print(arm, i, flush=True)
                del actor
                packed(root / (arm + ".json.gz"), seq)
                (root / (arm + "-partial.json.gz")).unlink(missing_ok=True)
        d = outcome(rows, runs)
    except (TimeoutError, MemoryError, InterruptedError) as e:
        d = {
            "passed": False,
            "status": "interrupted",
            "reason": str(e),
            "durable": {a: len(v) for a, v in runs.items()},
        }
        for arm, seq in runs.items():
            packed(root / (arm + "-partial.json.gz"), seq)
    finally:
        guard.close()
    save(root / "decision.json", d)


if __name__ == "__main__":
    run()
