"""Test one frozen online-trained policy on matched changed tables and the 220-item retention panel."""

import gc
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx

from ..evaluation.mechanism import packed, unpack
from ..io import digest, read, save
from ..reentry import ReentryAssistant
from ..relational_tools import tools
from ..tool_compare import DenseAssistant
from . import integrated_assistance as previous
from .core import Budget
from .cross_view_study import tokenizer, write_documents
from .file_evidence_environment import capture
from .linked_rollin import exact_sign_flip
from .native_function_study import evaluate, replay
from .registered import registered
from .table_policy_learning import NAME as LEARNED
from .table_reentry_fit import CODES, root_for
from .table_worlds import FILENAME, SOURCE, prepare_tables
from .table_worlds import score as table_score

NAME = "table-policy-gradient-01"
PARENT = Path("research/results") / LEARNED


def providers(w, folder):
    if w["family"] == "changed_table":
        return tools([folder / w["row_id"] / str(w["world"]) / FILENAME])
    return previous.providers(w, folder)


def score(tok, w, i, tr):
    if w["family"] == "changed_table":
        return table_score(w, tr)
    return previous.score(tok, w, i, tr)


def decision(panel, arms):
    base = previous.outcome(panel, arms)
    # Follow the original SQL and human thresholds and prior-success ledger. The rates and costs
    # in this invocation of previous.outcome include all 268 actual requests.
    groups = defaultdict(dict)
    variants = defaultdict(lambda: [0, 0])
    for w, c, b in zip(panel["rows"], arms["candidate"], arms["control"], strict=True):
        if w["family"] != "changed_table":
            continue
        x, y = (r["requests"][0]["score"]["success"] for r in (c, b))
        groups[w["cluster"]][w["world"]] = (x, y)
        variant = "original" if w["world"] == 0 else "altered"
        variants[variant][0] += x
        variants[variant][1] += y
    deltas = []
    counts = [0, 0]
    source = []
    for key, m in sorted(groups.items()):
        assert len(m) == 2 and 0 in m
        ok = [all(v[i] for v in m.values()) for i in (0, 1)]
        for i in (0, 1):
            counts[i] += ok[i]
        deltas.append(int(ok[0]) - int(ok[1]))
        source.append(
            {
                "question": key,
                "versions": {str(k): list(v) for k, v in m.items()},
                "whole": ok,
            }
        )
    p = exact_sign_flip(deltas)
    gates = base["gates"]
    gates.update(
        whole_tables=counts[0] >= 6 and counts[0] - counts[1] >= 5 and p <= 0.05,
        both_versions=all(a > b for a, b in variants.values()),
    )
    base.update(
        {
            "passed": all(gates.values()),
            "whole_table_counts": counts,
            "whole_table_p": p,
            "table_sources": source,
            "table_versions": dict(variants),
            "open_review": "Additional sealed reading with the original user questions and complete tables.",
        }
    )
    return base


def mounted(panel, folder):
    write_documents(panel, folder)
    prepare_tables(folder, read(SOURCE)["cohorts"]["dev"])


def verify(root, d):
    if d.get("status") == "interrupted":
        assert not d["passed"] and d["reason"]
        return
    s = registered(NAME)
    assert read(root / "registration.json") == s
    parent = read(root / "training-parent.json")
    assert parent["archive"] == digest(PARENT / "checksums.json") and parent[
        "weights"
    ] == digest(PARENT / "reentry.safetensors")
    tok = tokenizer(s)
    panel = read(s["material"])
    arms = {}
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        mounted(panel, folder)
        for a in previous.ARMS:
            arms[a] = unpack(root / (a + ".json.gz"))
            for w, r in zip(panel["rows"], arms[a], strict=True):
                (q,) = r["requests"]
                assert (
                    w["id"] == r["id"]
                    and q["trajectory"]["turns"][0]["prompt_ids"] == w["initial_ids"]
                )
                replay(
                    tok,
                    w,
                    providers(w, folder),
                    [{"role": "user", "content": w["questions"][0]}],
                    q,
                    s,
                    a,
                )
                assert score(tok, w, 0, q["trajectory"]) == q["score"]
    assert decision(panel, arms) == d


def run():
    s = registered(NAME)
    if not read(PARENT / "decision.json")["passed"]:
        raise ValueError(
            "The policy did not receive the registered execution-learning signal"
        )
    root = root_for(s)
    capture(
        root,
        (
            *CODES,
            "loopqwen/research/table_policy_learning.py",
            "loopqwen/research/table_policy_primary.py",
            "loopqwen/research/integrated_assistance.py",
            "loopqwen/research/native_function_study.py",
            "loopqwen/tool_compare.py",
        ),
    )
    guard = Budget(NAME, s["limits"])
    try:
        tok = tokenizer(s)
        panel = read(s["material"])
        arms = {}
        save(
            root / "training-parent.json",
            {
                "archive": digest(PARENT / "checksums.json"),
                "weights": digest(PARENT / "reentry.safetensors"),
            },
        )

        def base():
            actor = DenseAssistant(True, "qwen", s["checkpoint"])
            actor.native_plain = True
            return actor

        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            mounted(panel, folder)
            for side, make in (
                ("candidate", lambda: ReentryAssistant(s["checkpoint"], PARENT)),
                ("control", base),
            ):
                path = root / (side + ".json.gz")
                if not path.exists():
                    gc.collect()
                    mx.clear_cache()
                    rows = evaluate(
                        s,
                        panel,
                        side,
                        guard,
                        make,
                        lambda w: providers(w, folder),
                        scorer=lambda w, i, tr: score(tok, w, i, tr),
                    )
                    packed(path, rows)
                arms[side] = unpack(path)
        d = decision(panel, arms)
    except (TimeoutError, MemoryError, InterruptedError) as exc:
        d = {"passed": False, "status": "interrupted", "reason": str(exc)}
    finally:
        guard.close()
    save(root / "decision.json", d)


if __name__ == "__main__":
    run()
