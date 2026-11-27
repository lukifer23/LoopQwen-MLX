"""Evaluate one static recurrent checkpoint across transfer, prior skills and full-request cost."""

import csv
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from ..evaluation.mechanism import unpack
from ..io import digest, read, save
from ..relational_tools import tools as databases
from . import reasoned_evidence as pair
from .cross_view_study import tokenizer, write_documents
from .followup_study import providers as earlier_tools
from .followup_study import score as earlier_score
from .integrated_assistance_fit import batches, fit, positions
from .linked_rollin import exact_sign_flip
from .native_function_data import token_pair
from .native_function_study import executions, replay, summarize
from .registered import registered
from .task_merge import compile_consensus
from .task_merge_probe import measure
from .visit_residual_study import score as visit_score
from .visit_residual_study import tools_for as visit_tools

NAME = "integrated-assistance-01"
ARMS = ("candidate", "control")
PRIOR_IMPORT = ("archive", "human_first", "human_followup")
PRIOR_36 = ("agenda", "wheel", "retention", "file", "inline")


@lru_cache(maxsize=1)
def suites():
    from scripts.dialogue_database_suite import MANIFEST

    return read(MANIFEST)["by_schema"]


def providers(w, folder):
    if w["family"] == "database":
        from scripts.dialogue_database_sources import CACHE

        path = folder / (w["source"] + ".sqlite")
        if not path.is_file():
            with ZipFile(CACHE) as z:
                path.write_bytes(z.read(w["sqlite_member"]))
        assert digest(path) == w["sqlite_sha256"]
        return databases([path])
    if w["family"] == "open":
        return []
    if w["family"] in PRIOR_IMPORT:
        return earlier_tools(w, folder)
    return visit_tools(w, folder)


def db_score(w, tr):
    from scripts.dialogue_database_semantics import compare

    done = tr["status"] == "answered"
    events = executions({"turns": tr["turns"][:-1]}) if done else executions(tr)
    filename = w["source"] + ".sqlite"
    reads = [
        e
        for e in events
        if e["call"].get("name") == "database_query"
        and e["call"].get("arguments", {}).get("database_name") == filename
        and "rows" in e["observation"]
    ]
    read_correct = False
    result = {
        "passed": False,
        "matches": 0,
        "instances": len(suites()[w["source"]]["databases"]),
    }
    if reads:
        latest = reads[-1]
        observed = latest["observation"]
        read_correct = bool(
            observed["referenced_tables"]
            and not observed["truncated"]
            and observed["rows"] == [[w["expected_value"]]]
        )
        if read_correct:
            result = compare(
                latest["call"]["arguments"]["statement"],
                w["sql"],
                w["source"],
                w["turn"],
                suites()[w["source"]]["databases"],
            )
    txt = (tr["answer"] or "").strip()
    value = str(w["expected_value"])
    if isinstance(w["expected_value"], int):
        # An integer can also appear as the corresponding full decimal.
        pattern = r"(?<![\w.,-])" + re.escape(value) + r"(?:\.0+)?(?!\w|\.\d|,\d)"
    else:
        pattern = r"(?<!\w)" + re.escape(value) + r"(?!\w)"
    visible = bool(done and re.search(pattern, txt, re.IGNORECASE))
    return {
        "completed": done,
        "correct_answer": visible,
        "success": bool(visible and read_correct and result["passed"]),
        "schema_calls": sum(e["call"].get("name") == "database_schema" for e in events),
        "observed_queries": len(reads),
        "original_result": read_correct,
        "semantic": result,
    }


def score(tok, w, i, tr):
    family = w["family"]
    if family == "database":
        return db_score(w, tr)
    if family == "open":
        # Open-ended quality is scored only by the predeclared blinded reading.
        return {
            "completed": tr["status"] == "answered",
            "answer_quality": "requires masked review",
        }
    if family in PRIOR_IMPORT:
        return earlier_score(tok, w, i, tr)
    return visit_score(tok, w, i, tr)


def outcome(panel, arms):
    sides = {}
    wins = {}
    for arm, records in arms.items():
        assert [r["id"] for r in records] == [w["id"] for w in panel["rows"]]
        families = defaultdict(list)
        whole = defaultdict(lambda: defaultdict(list))
        turns = []
        byid = {}
        open_rows = []
        for w, r in zip(panel["rows"], records, strict=True):
            (q,) = r["requests"]
            turns.extend(q["trajectory"]["turns"])
            if w["family"] == "open":
                open_rows.append(q)
                continue
            families[w["family"]].append(r)
            byid[w["id"]] = q["score"]["success"]
            whole[w["family"]][w["cluster"]].append(q["score"]["success"])
        allq = [r["requests"][0] for r in records]
        closed = [r for r in records if r["family"] != "open"]
        sides[arm] = {
            "scored": summarize(closed),
            "families": {k: summarize(v) for k, v in families.items()},
            "whole": {
                k: sum(all(v) for v in groups.values()) for k, groups in whole.items()
            },
            "f1": {
                k: sum(r["requests"][0]["score"]["f1"] for r in families[k])
                / len(families[k])
                for k in ("human_first", "human_followup", "effects")
            },
            "all_cost": {
                "wall_s": sum(q["wall_s"] for q in allq),
                "first_s": sum(q["first_s"] for q in allq),
                "peak": max(q["peak"] for q in allq),
                "output_tokens": sum(len(t["token_ids"]) for t in turns),
                "opening_tps": sum(
                    len(t["prompt_ids"]) - t.get("reused_prompt_tokens", 0)
                    for t in turns
                )
                / sum(t["ttft_s"] for t in turns),
                "decode_tps": sum(len(t["token_ids"]) - 1 for t in turns)
                / sum(t["seconds"] - t["ttft_s"] for t in turns),
            },
            "open_completed": sum(q["score"]["completed"] for q in open_rows),
        }
        wins[arm] = byid
    a, b = [sides[k] for k in ARMS]
    difference = defaultdict(int)
    for w in panel["rows"]:
        if w["family"] == "database":
            difference[w["cluster"]] += int(wins["candidate"][w["id"]]) - int(
                wins["control"][w["id"]]
            )
    from scripts.followup_retention import NAMES

    with Path("paper/followup-retention.csv").open(newline="") as file:
        old_union = {
            r["id"] for r in csv.DictReader(file) if any(r[n] == "True" for n in NAMES)
        }
    previous = unpack(Path("research/results/followup-protected-01/candidate.json.gz"))
    known_software = {
        r["id"]
        for r in previous
        if r["family"] == "archive" and r["requests"][0]["score"]["success"]
    }
    assert len(known_software) == 20
    old_lost = sorted(i for i in old_union | known_software if not wins["candidate"][i])
    base_lost = sorted(
        k for k, v in wins["control"].items() if v and not wins["candidate"][k]
    )
    g = {
        "semantic_sql": sum(difference.values()) >= 8
        and exact_sign_flip(difference.values()) <= 0.05
        and a["whole"]["database"] >= max(4, b["whole"]["database"] + 2),
        "prior_skills": not old_lost,
        "prior_base": not base_lost,
        "human_reasoning": all(a["f1"][k] >= b["f1"][k] + 0.05 for k in a["f1"]),
        "all_families": all(
            a["families"][k]["grounded"] >= b["families"][k]["grounded"]
            for k in a["families"]
        ),
        "completion": all(
            a["families"][k]["completed"] >= b["families"][k]["completed"]
            for k in a["families"]
        )
        and a["open_completed"] >= b["open_completed"],
    }
    g.update(
        {k: a["all_cost"][k] < b["all_cost"][k] for k in ("first_s", "wall_s", "peak")}
    )
    g.update(
        {k: a["all_cost"][k] > b["all_cost"][k] for k in ("opening_tps", "decode_tps")}
    )
    return {
        "passed": all(g.values()),
        "gates": g,
        "arms": sides,
        "database_source_deltas": dict(difference),
        "database_p": exact_sign_flip(difference.values()),
        "prior_successes": len(old_union | known_software),
        "prior_lost": old_lost,
        "base_lost": base_lost,
        "additional_release_prerequisite": "Blinded original user assistance preference; independent fit seed and matched ordinary-depth and 128K trials if all initial gates pass.",
    }


def qualify(s, tok, root, guard):
    prior = read(Path("research/results") / s["deferred_after"] / "decision.json")
    if prior["passed"] or "arms" not in prior:
        raise InterruptedError("The preceding primary requires its own resolution")
    data = read(s["material"])
    for file, expected in s["sources"].items():
        assert digest(Path(file)) == expected
    assert (
        data["source"]["exact_train_held_prompt_overlaps"] == 0
        and len(data["rows"]) == 220
    )
    save(
        root / "source-qualification.json",
        {
            "material": digest(Path(s["material"])),
            "training": digest(Path(s["training"])),
            "files": s["sources"],
            "source": data["source"],
        },
    )
    guard.check()


def verify(root, d):
    if d.get("status") == "interrupted":
        assert not d["passed"] and d["reason"]
        return
    s = registered(NAME)
    assert s == read(root / "registration.json")
    tok = tokenizer(s)
    views = read(s["training"])["examples"]
    fitlog = read(root / "fit.json")
    assert (
        unpack(root / "training-views.json.gz") == views
        and fitlog["trainable"] == 802816
    )
    assert fitlog["weights"] == digest(root / "tool-adapters.safetensors")
    labels = [token_pair(tok, w)[1] for w in views]
    close = tok.convert_tokens_to_ids("</think>")
    visible = []
    for y in labels:
        j = y.index(close) + 1
        while j < len(y) - 1 and not tok.decode([y[j]]).strip():
            j += 1
        visible.append(j)
    draw = batches(views, s["fit_seed"])
    log = unpack(root / "optimizer-steps.json.gz")
    assert len(log) == fitlog["updates"] == s["updates"]
    n = 0
    for step, r in enumerate(log, 1):
        assert r["step"] == step and r["views"] == next(draw)
        assert len(r["losses"]) == 7 and all(
            math.isfinite(v) for v in r["losses"] + [r["gradient_norm"]]
        )
        for total, (second, first, ranking) in zip(
            r["losses"], r["components"], strict=True
        ):
            assert all(math.isfinite(v) for v in (second, first, ranking))
            assert math.isclose(
                total, second + 0.15 * ranking, rel_tol=2e-6, abs_tol=1e-6
            )
        for i, pos in zip(r["views"], r["positions"], strict=True):
            assert pos == positions(
                labels[i],
                visible[i],
                s["positions_per_view"],
                f"{s['fit_seed']}/{step}/{i}",
            )
            n += len(pos)
    assert n == fitlog["targets"]
    cm = read(root / "compiled.json")
    merge = read(root / "task-merge.json")
    assert cm["merge"] == digest(root / "task-merge.json") and merge["compiled"] == cm[
        "weights"
    ] == digest(root / "cross-view.safetensors")
    assert (
        merge["parents"] == s["merged_task_updates"]["parents"]
        and merge["new_expert"] == fitlog["weights"]
    )
    assert len(merge["projections"]) == 7 and all(
        r["nonzero"] > 0 and sum(r["participation"]) == r["elements"]
        for r in merge["projections"].values()
    )
    from .task_merge_probe import sample

    probes = unpack(root / "merged-training-visits.json.gz")
    for (kind, view), record in zip(sample(views), probes, strict=True):
        assert kind == record["group"] and view["id"] == record["view"]
        _, target = token_pair(tok, view)
        assert [target[j] for j in record["positions"]] == record["target"]
        for branch in ("first", "second"):
            assert len(record[branch]["top"]) == len(record["positions"])
            assert all(math.isfinite(x) and x >= 0 for x in record[branch]["nll"])
    panel = read(s["material"])
    runs = {a: unpack(root / (a + ".json.gz")) for a in ARMS}
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        write_documents(panel, folder)
        for arm, records in runs.items():
            for w, r in zip(panel["rows"], records, strict=True):
                (q,) = r["requests"]
                assert (
                    r["id"] == w["id"]
                    and q["trajectory"]["turns"][0]["prompt_ids"] == w["initial_ids"]
                )
                replay(
                    tok,
                    w,
                    providers(w, folder),
                    [{"role": "user", "content": w["questions"][0]}],
                    q,
                    s,
                    arm,
                )
                assert score(tok, w, 0, q["trajectory"]) == q["score"]
    assert outcome(panel, runs) == d


if __name__ == "__main__":
    pair.run(
        name=NAME,
        qualify=qualify,
        trainer=fit,
        compiler=compile_consensus,
        after_compile=measure,
        score_fn=score,
        tool_catalog=providers,
        decision_rule=outcome,
        extra_sources=(
            "loopqwen/research/integrated_assistance.py",
            "loopqwen/research/task_merge.py",
            "loopqwen/research/task_merge_probe.py",
            "loopqwen/research/integrated_assistance_fit.py",
            "scripts/dialogue_database_semantics.py",
            "loopqwen/relational_tools.py",
            "scripts/integrated_assistance_material.py",
        ),
    )
