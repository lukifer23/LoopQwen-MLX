"""Extract exact causal output-position states and read-only final attention memory."""

from pathlib import Path

import mlx.core as mx

from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import digest, read, save
from ..looped_model import LoopedModel
from ..recurrence import Span
from ..shared_query_assistant import from_backbone
from .conditional_features import feature_root
from .cross_view_study import tokenizer
from .file_evidence_environment import capture
from .host_pressure import budget, relinquish
from .native_function_data import token_pair
from .registered import registered
from .table_reentry_fit import root_for

FEATURES = "shared-query-features-01"
FIT = "shared-query-fit-01"
PRIMARY = "shared-query-transfer-01"
OLD = Path("research/results/conditional-features-01")
RESULTS = OLD.parent
SOURCES = (
    "loopqwen/shared_query.py",
    "loopqwen/shared_query_assistant.py",
    "loopqwen/conditional_reentry.py",
    "loopqwen/looped_model.py",
    "loopqwen/research/shared_query_features.py",
    "loopqwen/research/native_function_data.py",
    "scripts/shared_query_material.py",
    "uv.lock",
)


def predecessor():
    for parent in (
        "contrastive-table-features-01",
        "contrastive-language-features-01",
        "contrastive-world-fit-01",
        "contrastive-world-transfer-01",
    ):
        archive = RESULTS / parent
        if not (archive / "checksums.json").is_file():
            raise ValueError("The preceding experiment has not finished: " + parent)
        passed = read(archive / "decision.json")["passed"]
        evidence = {"stage": parent, "archive": digest(archive / "checksums.json")}
        if parent == "contrastive-world-transfer-01":
            complete = all(
                (archive / (arm + ".json.gz")).is_file()
                for arm in ("candidate", "control")
            )
            if complete:
                # Both source readings belong to every completed comparison, regardless
                # of its automatic result. A partial run has no complete pair to grade.
                root = Path("docs/measurements")
                service = root / "contrastive-services-reading.json"
                qualitative = root / "contrastive-qualitative-reading.json"
                masked = read(service)
                read(qualitative)
                evidence["source_readings"] = {
                    p.name: digest(p) for p in (service, qualitative)
                }
                passed = passed and masked["passed"]
            elif passed:
                raise ValueError(
                    "A passing transfer archive lacks complete generation streams"
                )
        if not passed:
            return evidence
    raise ValueError("Replicate the qualifying predecessor")


def arrays(net, p, y):
    pre, ordinary, k, v, pos, _ = net.encode(mx.array([p + y[:-1]]), start=len(p) - 1)
    assert (
        pos == len(p) - 1
        and ordinary.shape == pre.shape
        and ordinary.shape[1] == len(y)
    )
    assert k.shape == v.shape and k.shape[2] == len(p) + len(y) - 1
    return {
        "pre": pre,
        "ordinary": ordinary,
        "keys": k,
        "values": v,
        "labels": mx.array(y),
    }


def parity(net, p):
    # Use the complete natural opening, two physical inputs and stock Qwen reference.
    ids = mx.array([p])
    dense = LoopedModel(net.text, Span(len(net.text.layers) - 1, 1, 1))
    dense.eval()
    expected = dense.project(dense.hidden(ids)[:, -1:]).astype(mx.float32)
    full = net.project(net.last_hidden(ids, net.make_cache())).astype(mx.float32)
    mid = max(1, len(p) // 2)
    cache = net.make_cache()
    net.last_hidden(ids[:, :mid], cache)
    split = net.project(net.last_hidden(ids[:, mid:], cache)).astype(mx.float32)
    return {
        "tokens": len(p),
        "ordinary_error": mx.max(mx.abs(full - expected)).item(),
        "split_error": mx.max(mx.abs(full - split)).item(),
        "top": [mx.argmax(z).item() for z in (expected, full, split)],
        "attention_offsets": [
            int(slot.offset)
            for (i, _), slot in cache.slots.items()
            if not net.text.layers[i].is_linear
        ],
        "owners": len(cache.slots),
        "output_visits": 1 + net.revisits,
    }


def decision(s, idx, checks):
    gates = {
        "exposure": len(idx) == s["examples"]
        and sum(r["positions"] for r in idx) == s["target_positions"],
        "all_inputs": sum(r["opening"] for r in idx) == s["opening_positions"],
        "physical": all(
            r["owners"] == 24
            and r["output_visits"] == 3
            and r["attention_offsets"] == [r["tokens"]] * 6
            for r in checks
        ),
        "ordinary_and_cached": len(checks) == 3
        and all(
            len(set(r["top"])) == 1
            and r["ordinary_error"] <= 0.55
            and r["split_error"] <= 0.55
            for r in checks
        ),
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "examples": len(idx),
        "labels": sum(r["positions"] for r in idx),
        "file_bytes": sum(r["bytes"] for r in idx),
    }


def verify(root, d):
    s = registered(read(root / "registration.json")["id"])
    assert s == read(root / "registration.json")
    if d.get("status") == "interrupted" and not (root / "backbone.json").exists():
        assert (
            d["reason"] == "Host availability window expired"
            and d.get("durable", 0) == 0
        )
        return
    assert read(root / "backbone.json") == read(OLD / "backbone.json")
    if d.get("status") == "interrupted":
        assert not d["passed"]
        if (root / "partial-index.json.gz").exists():
            assert d["durable"] == len(unpack(root / "partial-index.json.gz"))
        else:
            assert d.get("durable", 0) == 0 and d["reason"]
        return
    idx = unpack(root / "feature-index.json.gz")
    assert decision(s, idx, read(root / "parity.json")) == d
    tok = tokenizer(s)
    for i, (r, w) in enumerate(zip(idx, unpack(Path(s["training"])), strict=True)):
        p, y = token_pair(tok, w)
        assert (r["id"], r["group"], r["opening"], r["positions"]) == (
            w["id"],
            w["group"],
            len(p),
            len(y),
        )
        assert (r["prefix"], r["labels"], r["file"]) == (
            fingerprint(p),
            fingerprint(y),
            f"{i:05}.safetensors",
        )
        assert len(r["sha256"]) == 64 and r["bytes"] > 0


def run(name=FEATURES):
    s = registered(name)
    if "precision_receipt" in s:
        receipt = Path(s["precision_receipt"])
        evidence = read(receipt)
        assert evidence["passed"] and len(evidence["records"]) == 12
        prior = {"precision_receipt": digest(receipt), "predecessor": s["predecessor"]}
    else:
        prior = predecessor()
    root = root_for(s)
    capture(
        root,
        (
            *SOURCES,
            "loopqwen/research/host_pressure.py",
            "scripts/repair_query_worker.py",
        )
        if s.get("max_pressure_restarts")
        else SOURCES,
    )
    save(root / "predecessor.json", prior)
    ident = read(OLD / "backbone.json")
    save(root / "backbone.json", ident)
    guard = budget(root, s)
    if guard is None:
        return
    folder = feature_root(s)
    journal = folder.parent / "partial-index.json.gz"
    idx = unpack(journal) if journal.exists() else []
    try:
        weights = from_backbone(s["checkpoint"], root, s, ident)
        tok = tokenizer(s)
        rows = unpack(Path(s["training"]))
        candidates = [
            min(
                (r for r in rows if r["group"] == g),
                key=lambda w: fingerprint([s["fit_seed"], w["id"]]),
            )
            for g in ("strategy", "report_action", "instruction")
        ]
        checks = [parity(weights, token_pair(tok, w)[0]) for w in candidates]
        save(root / "parity.json", checks)
        gates = decision(s, [], checks)["gates"]
        if not (gates["physical"] and gates["ordinary_and_cached"]):
            raise InterruptedError("Stock or cached Qwen parity failed")
        folder.mkdir(exist_ok=True, parents=True)
        for i, r in enumerate(idx):
            if r["id"] != rows[i]["id"] or digest(folder / r["file"]) != r["sha256"]:
                raise ValueError("A previously stored shared-query feature changed")
        for i in range(len(idx), len(rows)):
            w = rows[i]
            guard.check()
            p, y = token_pair(tok, w)
            data = arrays(weights, p, y)
            path = folder / f"{i:05}.safetensors"
            partial = path.with_suffix(".partial.safetensors")
            mx.save_safetensors(str(partial), data)
            partial.replace(path)
            idx.append(
                {
                    "id": w["id"],
                    "group": w["group"],
                    "opening": len(p),
                    "positions": len(y),
                    "prefix": fingerprint(p),
                    "labels": fingerprint(y),
                    "file": path.name,
                    "sha256": digest(path),
                    "bytes": path.stat().st_size,
                }
            )
            if len(idx) % 128 == 0:
                packed(journal, idx)
                print("Encoded", len(idx), "of", len(rows), flush=True)
        packed(root / "feature-index.json.gz", idx)
        d = decision(s, idx, checks)
    except (InterruptedError, TimeoutError, MemoryError) as error:
        if s.get("max_pressure_restarts"):
            packed(journal, idx)
            if relinquish(root, s, error, {"encoded": len(idx)}):
                return
        packed(root / "partial-index.json.gz", idx)
        d = {
            "passed": False,
            "status": "interrupted",
            "reason": str(error),
            "durable": len(idx),
        }
    finally:
        guard.close()
    save(root / "decision.json", d)


if __name__ == "__main__":
    run()
