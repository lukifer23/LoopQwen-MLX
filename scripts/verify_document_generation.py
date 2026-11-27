"""Compare the public shared-document route to fresh staged requests through each actual model."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import mlx.core as mx

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.shared_query_assistant import SharedQueryAssistant
from loopqwen.tool_compare import DenseAssistant
from scripts.content_read_material import DIR
from scripts.learner_prefix_material import FIT, RESULT
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

PLAN = Path("research/diagnostics/document-generation-parity.json")
OUT = Path("docs/measurements/document-generation-parity.json")


def requests():
    rows = unpack(DIR / "training.json.gz")
    by = {}
    for r in rows:
        if r["group"] == "calculate":
            by.setdefault(r["cluster"], []).append(r)
    key = min(by, key=lambda k: fingerprint(["document parity", k]))
    pair = by[key]
    assert len(pair) == 2
    return [r["messages"][0]["content"] for r in pair]


def run(freeze):
    qs = requests()
    plan = dict(
        requests=[fingerprint(q) for q in qs],
        fit=digest(RESULT / FIT / "checksums.json"),
        files={
            p: digest(Path(p))
            for p in (
                "loopqwen/offloaded_prefill.py",
                "loopqwen/tool_session.py",
                "loopqwen/tool_model.py",
                "loopqwen/evaluation/generation_eval.py",
                "scripts/verify_document_generation.py",
            )
        },
        protocol="Public request path on one real two-question training report. Stage their full common input once. On the frozen native two-read repair policy and the pinned untouched ordinary Base, compare eight unrestricted greedy outputs from each independent branch against a new complete staged prefill. Run question A, B, A. Require all 48 greedy predictions, exact repeatability, identical prompt IDs and physical history lengths at all six sites. Stop at 120 seconds or 3 GiB. Timing is an engineering observation on this short source.",
    )
    if freeze:
        persist(PLAN, plan)
        return
    committed(PLAN)
    assert read(PLAN) == plan
    started = monotonic()
    rows = []
    mx.reset_peak_memory()
    import gc

    for arm in ("loop", "base"):
        if arm == "loop":
            actor = SharedQueryAssistant(
                "exports/tool-language-packed-05", RESULT / FIT
            )
        else:
            actor = DenseAssistant(True, "qwen", "exports/tool-language-packed-05")
            actor.native_plain = True
            assert actor.language.base_source == "certified_release_text"
        with TemporaryDirectory() as root, TemporaryDirectory() as fresh:
            t = monotonic()
            doc = actor.prepare_document(qs, root)
            staging = monotonic() - t
            a = None
            for i in (0, 1, 0):
                opts = dict(
                    max_tokens=8,
                    max_turns=1,
                    record_kv_history=True,
                    prefill_cache_state=True,
                )
                ref = actor.respond(
                    qs[i], reuse_prefix=False, staged_context_dir=fresh, **opts
                )
                tr = actor.respond(qs[i], document_prefix=doc, **opts)
                tx = tr["turns"][0]
                rx = ref["turns"][0]
                eq = (
                    tx["token_ids"] == rx["token_ids"]
                    and tx["prompt_ids"] == rx["prompt_ids"]
                    and tx["kv_history"] == rx["kv_history"]
                    and tx["reused_prompt_tokens"] == len(doc.ids)
                )
                total = len(tx["prompt_ids"]) + len(tx["token_ids"]) - 1
                assert len(tx["kv_history"]) == 6 and all(
                    h["retained"] == h["offset"] == total for h in tx["kv_history"]
                )
                if i == 0 and a is not None:
                    eq = eq and a == tx["token_ids"]
                if i == 0:
                    a = tx["token_ids"]
                rows.append(
                    dict(
                        arm=arm,
                        question=i,
                        equal=eq,
                        output=tx["token_ids"],
                        reference=rx["token_ids"],
                        opening=len(tx["prompt_ids"]),
                        reused=len(doc.ids),
                        new=tx["new_prompt_tokens"],
                        staging_s=staging,
                        branch_s=tr["seconds"],
                        fresh_s=ref["seconds"],
                    )
                )
                if monotonic() - started > 120 or mx.get_peak_memory() > 3 * 2**30:
                    raise RuntimeError("Document parity resource cap")
            del doc
        del actor
        gc.collect()
        mx.clear_cache()
    out = dict(
        passed=all(r["equal"] for r in rows),
        rows=rows,
        plan=digest(PLAN),
        seconds=monotonic() - started,
        peak_bytes=mx.get_peak_memory(),
    )
    persist(OUT, out)
    print(out["passed"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("register", "measure"))
    run(p.parse_args().action == "register")
