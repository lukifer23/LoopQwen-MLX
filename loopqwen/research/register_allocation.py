"""Paired full-history prefill comparisons against the stock language decoder."""

import argparse
import gc
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from ..deployment import load_trained
from ..evaluation.answers import numeric
from ..evaluation.language7 import interpret
from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import digest, read, save
from ..model_loader import local_snapshot
from .balanced_register import flatten
from .core import Budget
from .ledger_curriculum import measure
from .registered import registered


def packed_evidence(s, consumed, *, capacity=None):
    # Pinned Qwen3.5-0.8B has two KV heads, each with 256 channels.
    # Two arrays (key and value), one byte per channel and two BF16 affine
    # parameters per 64-channel group. These are live tensor sizes.
    code_bytes = consumed * 2 * 2 * 256
    if capacity is not None and not (
        s["capacity"] == capacity
        and 512 <= s["direct_queries"] <= s["dense_queries"]
        and 1 <= s["max_tile_queries"] <= 8
        and s["direct_tiles"] * 8 >= s["direct_queries"]
    ):
        return False
    return (
        s["bits"] == 8
        and s["group_size"] == 64
        and s["code_bytes"] == code_bytes
        and s["affine_bytes"] == code_bytes // 16
    )


def direct_outcome(rows, items, costs, policy, cap):
    evidence = True
    ordinary = {(i, 0) for i in (3, 7, 11, 15, 19, 23)}
    recurrent = ordinary | {(11, 1), (15, 1)}
    for side, arm in rows.items():
        for w, r in zip(items, arm, strict=True):
            n = len(w["prompt_ids"])
            consumed = n + len(r["token_ids"]) - 1
            sites = r.get("kv_history", [])
            evidence &= (
                r["input_sha256"] == fingerprint(w["prompt_ids"])
                and n >= 131072
                and n == r["prompt_tokens"]
                and {(s["layer"], s["turn"]) for s in sites}
                == (ordinary if side == "base" else recurrent)
                and all(s["offset"] == s["retained"] == consumed for s in sites)
            )
            if side == policy:
                section = r.get("attention_memory")
                selected = r["page_sites"]
                evidence &= (
                    section is not None
                    and r.get("context_route") == "register"
                    and section["start"]
                    <= w["gold_start"]
                    <= w["gold_stop"]
                    <= section["stop"]
                    and {(s["layer"], s["turn"]) for s in selected} == recurrent
                    and all(
                        s["sparse_blocks"] * 128 + s["dense_queries"] == consumed
                        and s["dense_queries"] >= 512
                        and s["sparse_blocks"] > 0
                        and s["offset"] == s["retained"] == consumed
                        and s["possible"] == consumed * (consumed + 1) // 2
                        and 0 < s["attended"] < s["possible"]
                        and (
                            policy == "pages"
                            or packed_evidence(
                                s,
                                consumed,
                                capacity=n + cap - 1
                                if policy == "direct_pages"
                                else None,
                            )
                        )
                        for s in selected
                    )
                )
    a, b = costs[policy], costs["base"]
    useful = bool(
        evidence
        and a["correct"] == 2 > b["correct"]
        and a["ttft"] < b["ttft"]
        and a["seconds"] < b["seconds"]
    )
    return {
        "passed": useful and a["peak"] <= b["peak"],
        "accuracy_and_time": useful,
        "full_history": bool(evidence),
        "metrics": costs,
        "items": len(items),
    }


def outcome(spec, rows, items):
    costs = {
        name: {
            "ttft": sum(r["ttft_s"] for r in rs),
            "seconds": sum(r["elapsed_s"] for r in rs),
            "peak": max(r["peak_bytes"] for r in rs),
            "correct": sum(r["correct"] for r in rs),
        }
        for name, rs in rows.items()
    }
    policy = spec.get("attention_policy", "reserved")
    control = spec.get("control", "stock")
    if control == "base":
        decision = direct_outcome(rows, items, costs, policy, spec["max_tokens"])
        if "fused_export" in spec:
            from .full_decode import performance

            for arm in rows.values():
                for r in arm:
                    duration = r["elapsed_s"] - r["ttft_s"]
                    further = len(r["token_ids"]) - 1
                    if not (
                        duration > 0
                        and further > 0
                        and r["decode_tps"] == further / duration
                        and r["prefill_tps"] == r["prompt_tokens"] / r["ttft_s"]
                    ):
                        raise ValueError(
                            "Per-request throughput differs from its timed token count"
                        )
            measured = {side: performance(arm) for side, arm in rows.items()}
            fused, base = measured[policy], measured[control]
            sites = all(
                s.get("fused_queries") == len(r["token_ids"]) - 1
                for r in rows[policy]
                for s in r["page_sites"]
            )
            decision.update(
                throughput=measured,
                fused_at_every_site=sites,
                passed=decision["passed"]
                and sites
                and fused["peak"] < base["peak"]
                and all(fused[k] > base[k] for k in ("prefill_tps", "decode_tps")),
            )
        return decision
    c, b = costs[policy], costs[control]
    exact = True
    for a, o, w in zip(rows[policy], rows[control], items, strict=True):
        sec = a["attention_memory"]
        exact &= (
            a["input_sha256"] == o["input_sha256"] == fingerprint(w["prompt_ids"])
            and a["token_ids"] == o["token_ids"]
            and sec == o["attention_memory"]
            and sec["start"] <= w["gold_start"] <= w["gold_stop"] <= sec["stop"]
            and a["context_route"] == o["context_route"] == "register"
        )
        expected = {
            (3, 0),
            (7, 0),
            (11, 0),
            (11, 1),
            (15, 0),
            (15, 1),
            (19, 0),
            (23, 0),
        }
        consumed = len(w["prompt_ids"]) + len(a["token_ids"]) - 1
        if policy in ("pages", "packed_pages", "direct_pages"):
            if policy in ("packed_pages", "direct_pages"):
                capacity = (
                    len(w["prompt_ids"]) + spec["max_tokens"] - 1
                    if policy == "direct_pages"
                    else None
                )
                exact &= all(
                    packed_evidence(s, consumed, capacity=capacity)
                    for s in a["page_sites"]
                )
                exact &= all(
                    s["sparse_blocks"] > 0 and s["retained"] == consumed
                    for s in o["page_sites"]
                )
            sites = a["page_sites"]
            exact &= (
                {(s["layer"], s["turn"]) for s in sites} == expected
                and all(
                    s["sparse_blocks"] > 0
                    and s["retained"] == s["offset"] == consumed
                    and s["dense_queries"] + s["sparse_blocks"] * 128 == consumed
                    and s["dense_queries"] >= 512
                    and s["possible"] == consumed * (consumed + 1) // 2
                    and 0 < s["attended"] < s["possible"]
                    for s in sites
                )
                and all(
                    {(s["layer"], s["turn"]) for s in r["kv_history"]} == expected
                    and all(
                        s["retained"] == s["offset"] == consumed
                        for s in r["kv_history"]
                    )
                    for r in (a, o)
                )
            )
        else:
            sites = a["reserved_sites"]
            exact &= {(s["layer"], s["turn"]) for s in sites} == expected and all(
                s["capacity"] == len(w["prompt_ids"]) + spec["max_tokens"] - 1
                and s["offset"] == consumed
                for s in sites
            )
    return {
        "passed": bool(
            exact
            and c["correct"] == len(items)
            and c["ttft"] <= spec["ratio"] * b["ttft"]
            and c["seconds"] <= spec.get("elapsed_ratio", 1) * b["seconds"]
            and c["peak"] <= spec.get("peak_ratio", 1) * b["peak"]
        ),
        "exact": exact,
        "metrics": costs,
        "items": len(items),
    }


def verify(root, decision):
    spec = read(root / "registration.json")
    items = flatten(spec)
    tok = load_tokenizer(local_snapshot())
    term = {*tok.eos_token_ids, tok.convert_tokens_to_ids("<|im_end|>")}
    rows = {
        k: unpack(root / (k + ".json.gz"))
        for k in (
            spec.get("attention_policy", "reserved"),
            spec.get("control", "stock"),
        )
    }
    if spec.get("control") == "base":
        from .counterfactual_memory import material

        assert read(spec["material"]) == [
            material(tok, {"seed": spec["seed"], "minimum_length": 131072})
        ]
    for rs in rows.values():
        for row, w in zip(rs, items, strict=True):
            ids = row["token_ids"]
            parsed = interpret(
                ids,
                tok.convert_tokens_to_ids("<think>"),
                tok.convert_tokens_to_ids("</think>"),
                term,
                tok,
            )
            answer = numeric(parsed["visible"]) if ids[-1] in term else None
            assert answer == row["answer"] and (answer == w["gold"]) == row["correct"]
            assert row["input_sha256"] == fingerprint(w["prompt_ids"]) and row[
                "prompt_tokens"
            ] == len(w["prompt_ids"])
    assert outcome(spec, rows, items) == decision


def run(name):
    spec = registered(name)
    root = Path("outputs/research") / name
    if (root / "decision.json").exists():
        raise FileExistsError(root)
    parent = Path(spec["requires"])
    assert (
        digest(parent / "checksums.json") == spec["prerequisite_sha256"]
        and read(parent / "decision.json")["passed"]
    )
    if "fused_export" in spec:
        release = Path(spec["release"])
        assert digest(release / "checksums.json") == spec["release_sha256"]
        files = read(release / "export-files.json")
        assert all(
            digest(Path(spec["fused_export"]) / k) == v for k, v in files.items()
        )
    budget = Budget(name, spec["limits"])
    policy = spec.get("attention_policy", "reserved")
    control = spec.get("control", "stock")
    items = flatten(spec)
    try:
        from .paired_progress import resume

        rows = (
            resume(root, spec, items, (policy, control))
            if control in ("base", "pages")
            else {policy: [], control: []}
        )
        for i, item in enumerate(items):
            for side in (policy, control) if i % 2 == 0 else (control, policy):
                if len(rows[side]) > i:
                    continue
                if side == "base":
                    from ..evaluation.language7 import base

                    model, tok, _ = base()
                elif "fused_export" in spec:
                    from ..export_runtime import load_export

                    model, tok = load_export(Path(spec["fused_export"]))
                    object.__setattr__(model, "fused_packed_decode", True)
                else:
                    model, tok, _ = load_trained(spec["checkpoint"])
                if (
                    side in ("pages", "packed_pages", "direct_pages")
                    and "fused_export" not in spec
                ):
                    from ..page_prefill import attach

                    slots = attach(
                        model,
                        len(item["prompt_ids"]),
                        packed=side == "packed_pages",
                        direct_capacity=len(item["prompt_ids"]) + spec["max_tokens"] - 1
                        if side == "direct_pages"
                        else None,
                    )
                mx.reset_peak_memory()
                record = measure(
                    model,
                    tok,
                    item,
                    spec["max_tokens"],
                    budget,
                    reserve_full_kv=side == "reserved",
                    record_kv_history=policy
                    in ("pages", "packed_pages", "direct_pages"),
                )
                if (
                    side in ("pages", "packed_pages", "direct_pages")
                    and "fused_export" not in spec
                ):
                    record["page_sites"] = [
                        dict(layer=i, turn=p, **slot.report())
                        for (i, p), slot in slots.items()
                    ]
                    del slots
                rows[side].append(record)
                if control in ("base", "pages"):
                    packed(root / (side + ".json.gz"), rows[side])
                print(
                    i,
                    side,
                    record["correct"],
                    record["ttft_s"],
                    record["peak_bytes"],
                    flush=True,
                )
                del model, tok
                gc.collect()
                mx.clear_cache()
        for side, rs in rows.items():
            packed(root / (side + ".json.gz"), rs)
        save(root / "decision.json", outcome(spec, rows, items))
    finally:
        budget.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    run(p.parse_args().name)
