"""One resident instruction backbone for randomly interleaved reasoning and tool turns."""

import argparse
import gc
from collections import Counter
from pathlib import Path
from time import perf_counter

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.utils import load_tokenizer

from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import digest, read, save
from ..model_loader import local_snapshot
from ..shared_session import SharedAssistant
from ..tool_compare import DenseAssistant
from ..tool_session import initial_messages, prompt
from ..unified_session import UnifiedAssistant
from .core import Budget
from .evidence_gate_fit import fit_gate
from .folded_system import family_scores, judge, old_scores, tools_for
from .mixed_field import replay_exchange
from .registered import registered
from .tool_fit import fitted, restore


def sources(spec):
    field = read(spec["field_data"])
    panel = read(spec["material"])
    assert digest(Path(spec["field_data"])) == panel["field_sha256"]
    return field, panel["rows"]


def exported_arrays(net):
    return {id(v): v.nbytes for _, v in tree_flatten(net.parameters())}


def native_receipt(actor, row):
    actor._load_language()
    a, b = actor.language, actor._tool_model
    for plain, typed in zip(a.text.layers[:23], b.text.layers[:23], strict=True):
        assert all(
            x is y
            for (_, x), (_, y) in zip(
                tree_flatten(plain.parameters()),
                tree_flatten(typed.parameters()),
                strict=True,
            )
        )
    first, second = exported_arrays(a), exported_arrays(b)
    ids = row["initial_ids"][-32:]
    gate_weights = (
        {} if isinstance(actor, UnifiedAssistant) else exported_arrays(actor.gate)
    )
    receipt = {
        "shared_bytes": sum(first[k] for k in first.keys() & second.keys()),
        "combined_bytes": sum((first | second | gate_weights).values()),
        "ids": ids,
        "views": {},
    }
    if isinstance(actor, UnifiedAssistant):
        receipt["same_model_object"] = a is b
    for label, net in (("reasoning", a), ("tool", b)):
        cache = net.make_cache()
        assert (
            len(net.text.layers) == 24
            and len(net.sites) == 25
            and cache.slots[23, 0] is not cache.slots[23, 1]
        )
        full = net.project(net.hidden(mx.array([ids]))[:, -1:])
        mx.eval(net.hidden(mx.array([ids[:-1]]), cache))
        one = net.project(net.hidden(mx.array([ids[-1:]]), cache))
        receipt["views"][label] = {
            "weights": sum(exported_arrays(net).values()),
            "layers": len(net.text.layers),
            "calls": len(net.sites),
            "full": full.argmax().item(),
            "incremental": one.argmax().item(),
            "max_logit_error": mx.abs(full.astype(mx.float32) - one.astype(mx.float32))
            .max()
            .item(),
            "offsets": [cache.slots[23, j].offset for j in (0, 1)],
        }
    return receipt


def decision(spec, rows, panel, receipt, router):
    groups, lost = family_scores(rows, panel, old_scores(spec))
    a, b = groups["all"]["candidate"], groups["all"]["control"]
    fresh = groups["fresh"]
    routes = Counter(r["trajectory"]["short_policy_route"] for r in rows["candidate"])
    accuracy = {}
    if spec.get("unified"):
        policy_ok = (
            receipt["same_model_object"]
            and receipt["shared_bytes"] == receipt["combined_bytes"]
        )
        required_gains = ("math", "judgment", "user_extraction")
    else:
        for external in (False, True):
            vals = [r for r in router["validation"] if r["external"] == external]
            accuracy[str(external)] = sum(
                (r["p"] >= spec["plain_cutoff"]) == external for r in vals
            ) / len(vals)
        policy_ok = min(accuracy.values()) >= 0.85
        required_gains = ("math", "judgment")
    passed = (
        a["success"] >= b["success"] + spec["gain"]
        and groups["all"]["p"] <= 0.05
        and groups["all"]["cluster_p"] <= 0.05
        and fresh["gains"] - fresh["losses"] >= spec["fresh_gain"]
        and all(
            g["candidate"][k] >= g["control"][k]
            for g in groups.values()
            for k in ("success", "completed", "answer")
        )
        and all(
            groups[k]["candidate"]["success"] >= groups[k]["control"]["success"] + 1
            for k in required_gains
        )
        and a["grounded"] >= b["grounded"] + 12
        and not any(lost.values())
        and all(a[k] < b[k] for k in ("seconds", "first", "peak"))
        and policy_ok
        and receipt["shared_bytes"]
        >= 0.98 * min(v["weights"] for v in receipt["views"].values())
        and all(
            v["full"] == v["incremental"]
            and v["max_logit_error"] <= 0.6
            and v["offsets"] == [32, 32]
            for v in receipt["views"].values()
        )
    )
    return {
        "passed": bool(passed),
        "groups": groups,
        "historically_correct_now_lost": lost,
        "routes": dict(routes),
        "router_validation": accuracy,
        "native": receipt,
    }


def verify(root, d):
    spec = read(root / "registration.json")
    field, panel = sources(spec)
    training = read(spec["training"])
    meta = read(root / "fit.json")
    assert (
        meta["passes"] == 2
        and meta["updates"] == spec["updates"]
        and meta["training"] == digest(Path(spec["training"]))
        and meta["turns"] == len(training["examples"])
    )
    assert meta["weights"] == digest(root / "tool-adapters.safetensors")
    router = None
    if spec.get("unified"):
        from .unified_session_fit import verify_sources

        verify_sources(spec, training, panel)
        assert meta["source_counts"] == Counter(
            w["stratum"] for w in training["examples"]
        )
        assert meta["contrast_examples"] > 0 and meta["distinct_training_rows"] > 0
        assert meta["sampled_by_source"] == {g: spec["updates"] for g in spec["strata"]}
        assert digest(Path(spec["material"])) == spec["panel_sha256"]
        assert meta["training"] == spec["training_sha256"]
        assert (
            len(panel) == 144
            and sum(w["family"] == "user_extraction" for w in panel) == 24
        )
    else:
        router = read(root / "router.json")
        assert (
            router["weights"] == digest(root / "router.safetensors")
            and router["updates"] == spec["gate_updates"]
        )
        assert [
            (w["id"], w["external"])
            for w in training["router"]
            if w["split"] == "validation"
        ] == [(w["id"], w["external"]) for w in router["validation"]]
        assert all(0 <= w["p"] <= 1 for w in router["validation"])
    tok = load_tokenizer(local_snapshot())
    if spec.get("unified"):
        from .unified_session_fit import verify_sampling

        verify_sampling(spec, training["examples"], meta, tok)
    failed_input = d.get("status") == "unequal_initial_ids"
    data = {}
    for side in ("candidate",) if failed_input else ("candidate", "control"):
        arm = unpack(root / (side + ".json.gz"))
        for w, r in zip(panel[: len(arm)] if failed_input else panel, arm, strict=True):
            tr = r["trajectory"]
            tools = tools_for(w, field, True)
            user = w.get("user_request", w["question"])
            msgs = (
                initial_messages(user, tools)
                if tools
                else [{"role": "user", "content": user}]
            )
            assert (
                w["initial_ids"]
                == tr["turns"][0]["prompt_ids"]
                == tok.encode(prompt(tok, msgs), add_special_tokens=False)
            )
            assert w["id"] == r["id"] and r["input_sha256"] == fingerprint(
                w["initial_ids"]
            )
            replay_exchange(tok, tools, msgs, tr, spec)
            assert (
                r["score"] == judge(w, tr, tok)
                and 0 < r["first_token_s"] <= r["wall_s"]
                and r["wall_s"] >= tr["seconds"]
                and r["peak"] > 0
            )
            sites = {(i, 0) for i in (3, 7, 11, 15, 19, 23)}
            route = tr.get("short_policy_route")
            if side == "candidate" and route in ("reasoning", "tool", "unified"):
                sites.add((23, 1))
            for t in tr["turns"]:
                kv = t["kv_history"]
                if route not in ("parent_evidence", "parent_long"):
                    assert {(v["layer"], v["turn"]) for v in kv} == sites
                assert all(
                    v["offset"]
                    == v["retained"]
                    == len(t["prompt_ids"]) + len(t["token_ids"]) - 1
                    for v in kv
                )
            if spec.get("unified") and side == "candidate":
                assert route in (
                    ("unified",)
                    if tools
                    else ("unified", "parent_long", "parent_evidence")
                )
            elif side == "candidate" and tools:
                assert 0 <= tr["external_probability"] <= 1 and (
                    tr["short_policy_route"] == "reasoning"
                ) == (tr["external_probability"] < spec["plain_cutoff"])
        data[side] = arm
    if failed_input:
        from .tokenizer_parity import audit, stopped

        report = read(root / "tokenizer-parity.json")
        assert (
            audit(spec, field, panel) == report
            and not (root / "control.json.gz").exists()
        )
        assert d == stopped(panel, data["candidate"], report)
        return
    receipt = read(root / "native.json")
    assert receipt["ids"] == panel[0]["initial_ids"][-32:] and all(
        v["calls"] == 25 and v["layers"] == 24 for v in receipt["views"].values()
    )
    expected = decision(spec, data, panel, receipt, router)
    assert d == qualified_replay(spec, root, data, expected)


def qualified_replay(spec, root, data, outcome):
    if "frozen_fit" not in spec:
        return outcome
    from scripts.field_cli_replay import identical

    fitted = Path(spec["frozen_fit"])
    assert digest(fitted / "checksums.json") == spec["frozen_inventory"]
    assert digest(root / "fit.json") == digest(fitted / "fit.json")
    assert digest(root / "tool-adapters.safetensors") == spec["frozen_weights"]
    prior = unpack(fitted / "candidate.json.gz")
    matches = sum(
        a["id"] == b["id"] and identical(a["trajectory"], b["trajectory"])
        for a, b in zip(prior, data["candidate"][: len(prior)], strict=True)
    )
    return {
        **outcome,
        "passed": outcome["passed"] and matches == len(prior),
        "earlier_identical": matches,
        "earlier_enrolled": len(prior),
    }


def run(name):
    spec = registered(name)
    checkpoint = Path(spec["checkpoint"])
    root = Path("outputs/research") / name
    root.mkdir(parents=True, exist_ok=True)
    if (root / "decision.json").is_file():
        raise FileExistsError(root)
    release = Path(spec["release"])
    assert (
        read(release / "decision.json")["passed"]
        and digest(release / "checksums.json") == spec["release_sha256"]
    )
    assert (
        digest(checkpoint / "tool-export-files.json")
        == digest(release / "export-files.json")
        == spec["bundle_inventory"]
    )
    assert all(
        digest(checkpoint / path) == sha
        for path, sha in read(release / "export-files.json").items()
    )
    field, panel = sources(spec)
    from ..package_tools import snapshot

    assert snapshot() == field["environment"] and all(
        digest(Path(p)) == v["sha256"] for p, v in field["files"].items()
    )
    save(root / "registration.json", spec)
    budget = Budget(name, spec["limits"])
    data = {}
    try:
        if spec.get("unified"):
            if "frozen_fit" in spec:
                from shutil import copyfile

                fitted = Path(spec["frozen_fit"])
                assert digest(fitted / "checksums.json") == spec["frozen_inventory"]
                assert (
                    digest(fitted / "tool-adapters.safetensors")
                    == spec["frozen_weights"]
                )
                for key in ("fit.json", "tool-adapters.safetensors"):
                    copyfile(fitted / key, root / key)
            elif not (root / "fit.json").is_file():
                from .unified_session_fit import train

                train(spec, root, budget)
        elif not (root / "router.json").is_file():
            model, tok = (
                restore(root)
                if (root / "fit.json").exists()
                else fitted("train_loop", spec, root, budget)
            )
            fit_gate(model, tok, read(spec["training"])["router"], spec, root, budget)
            del model, tok
        for side in ("candidate", "control"):
            dest = root / (side + ".json.gz")
            if dest.is_file() and len(unpack(dest)) == len(panel):
                data[side] = unpack(dest)
                continue
            actor = None
            arm = []
            gc.collect()
            mx.clear_cache()
            for w in panel:
                tools = tools_for(w, field)
                mx.synchronize()
                mx.reset_peak_memory()
                at = perf_counter()
                if actor is None:
                    if side == "control":
                        actor = DenseAssistant(False)
                    else:
                        service = (
                            UnifiedAssistant if spec.get("unified") else SharedAssistant
                        )
                        actor = service(checkpoint, root)
                stamps = []

                expected_tokens = len(w["initial_ids"])

                def progress(n, expected=expected_tokens, start=at, out=stamps):
                    if n == expected and not out:
                        out.append(perf_counter() - start)
                    budget.check()

                tr = actor.respond(
                    w.get("user_request", w["question"]),
                    tools,
                    max_tokens=spec["max_tokens"],
                    max_turns=spec["max_turns"],
                    reuse_prefix=False,
                    record_kv_history=True,
                    progress=progress,
                )
                mx.synchronize()
                wall = perf_counter() - at
                assert (
                    len(stamps) == 1
                    and tr["turns"][0]["prompt_ids"] == w["initial_ids"]
                )
                arm.append(
                    {
                        "id": w["id"],
                        "input_sha256": fingerprint(w["initial_ids"]),
                        "trajectory": tr,
                        "score": judge(w, tr, actor.tokenizer),
                        "first_token_s": stamps[0],
                        "wall_s": wall,
                        "peak": mx.get_peak_memory(),
                    }
                )
                if len(arm) % 12 == 0:
                    packed(dest, arm)
                    print(
                        side,
                        len(arm),
                        sum(bool(r["score"]["success"]) for r in arm),
                        flush=True,
                    )
            if side == "candidate":
                save(root / "native.json", native_receipt(actor, panel[0]))
            data[side] = arm
            del actor
        outcome = decision(
            spec,
            data,
            panel,
            read(root / "native.json"),
            None if spec.get("unified") else read(root / "router.json"),
        )
        save(root / "decision.json", qualified_replay(spec, root, data, outcome))
    finally:
        budget.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    run(p.parse_args().name)
