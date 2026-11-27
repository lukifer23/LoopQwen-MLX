"""Independently regenerate final-layer training memory and replay learned conclusions on Metal."""

import argparse
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import mlx.core as mx

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.conditional_features import feature_root
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from loopqwen.research.registered import registered
from loopqwen.research.shared_query_features import FEATURES, FIT, OLD, RESULTS, arrays
from loopqwen.research.shared_query_fit import decision, probe, verify
from loopqwen.shared_query_assistant import SharedQueryAssistant, from_backbone
from loopqwen.worker import acquire


def check(name, restore):
    acquire()
    started = perf_counter()
    mx.reset_peak_memory()
    s = registered(name)
    root = RESULTS / name
    fs = registered(s.get("feature_stage", name if "examples" in s else FEATURES))
    features = RESULTS / fs["id"]
    outcome = read(root / "decision.json")
    if "expected_positions" in s:
        if outcome.get("status") == "interrupted":
            raise ValueError(
                "A completed optimizer schedule is required for weight replay"
            )
        verify(root, outcome)
    elif not outcome["passed"]:
        raise ValueError("Feature extraction must be complete for source-memory replay")
    folder = feature_root(fs)
    folder.mkdir(exist_ok=True, parents=True)
    idx = unpack(features / "feature-index.json.gz")
    rows = unpack(Path(s["training"]))
    tok = tokenizer(s)
    net = from_backbone(s["checkpoint"], root, s, read(OLD / "backbone.json"))
    pick = {
        min(
            (i for i, r in enumerate(idx) if r["group"] == g),
            key=lambda i: fingerprint(["cold native", idx[i]["id"]]),
        )
        for g in (s.get("cold_groups") or ("strategy", "report_action", "full_rows"))
    }
    rebuilt = 0
    verified = []
    with TemporaryDirectory() as tmp:
        for i, (r, w) in enumerate(zip(idx, rows, strict=True)):
            path = folder / r["file"]
            if i in pick or restore and not path.exists():
                p, y = token_pair(tok, w)
                assert r["prefix"] == fingerprint(p) and r["labels"] == fingerprint(y)
                dest = Path(tmp) / r["file"]
                mx.save_safetensors(str(dest), arrays(net, p, y))
                assert digest(dest) == r["sha256"]
                if not path.exists():
                    if not restore:
                        raise FileNotFoundError(str(path))
                    # Move across filesystems through a sibling temporary file.
                    import shutil

                    copy = path.with_suffix(".restoring")
                    shutil.copyfile(dest, copy)
                    copy.replace(path)
                    rebuilt += 1
                if i in pick:
                    verified.append(r["id"])
            assert digest(path) == r["sha256"]
    report = {
        "features": digest(features / "checksums.json"),
        "checked_files": len(idx),
        "independent_regenerations": verified,
    }
    if "expected_positions" in s:
        # This process has no reference to the optimizer or training graph.
        evidence = unpack(Path(s["evidence"]))
        before = probe(net, idx, folder, evidence, s)
        del net
        mx.clear_cache()
        model = SharedQueryAssistant(s["checkpoint"], root)._tool_model
        native = {
            "before": before,
            "after": probe(model, idx, folder, evidence, s),
            "first_visit": probe(model, idx, folder, evidence, s, first_only=True),
        }
        saved = read(root / "probes.json")
        history = unpack(root / "training-log.json.gz")
        expected_ids = Counter(r["id"] for r in idx)
        assert all(n == 1 for n in expected_ids.values())
        assert {h["epoch"] for h in history} == set(range(s["epochs"]))
        exposure = []
        for epoch in range(s["epochs"]):
            updates = [h for h in history if h["epoch"] == epoch]
            observed = Counter(k for h in updates for k in h["rows"])
            assert observed == expected_ids
            exposure.append(
                {
                    "epoch": epoch + 1,
                    "unique_turns": len(observed),
                    "updates": len(updates),
                    "positions": sum(h["positions"] for h in updates),
                    "citations": sum(h["citations"] for h in updates),
                }
            )
        errors = {}
        for state, records in native.items():
            pairs = list(zip(records, saved[state], strict=True))
            assert all(
                r[k] == a[k]
                for r, a in pairs
                for k in ("id", "group", "positions", "citations")
            )
            errors[state] = {
                m: max(abs(r[m] - a[m]) for r, a in pairs)
                for m in ("ce", "ordinary_ce", "kl", "citation_nll")
            }
            assert max(errors[state].values()) <= 1e-5
        replay = decision(
            s,
            history,
            native["before"],
            native["after"],
            native["first_visit"],
        )
        assert (
            replay["gates"] == outcome["gates"]
            and replay["passed"] == outcome["passed"]
        )
        report.update(
            fit=digest(root / "checksums.json"),
            probe_cases=len(native["after"]),
            trained_positions=sum(r["positions"] for r in native["after"]),
            max_errors=errors["after"],
            max_errors_by_state=errors,
            reproduced_gates=replay["gates"],
            reproduced_pass=replay["passed"],
            independently_counted_epochs=exposure,
        )
    report.update(
        native_seconds=perf_counter() - started, peak_bytes=mx.get_peak_memory()
    )
    out = Path("docs/measurements") / (name + "-cold.json")
    save(out, report)
    print(
        len(idx),
        "verified,",
        rebuilt,
        "restored;",
        len(verified),
        "original native arrays reproduced",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage",
        choices=(FEATURES, FIT, "repair-query-features-01", "repair-query-fit-01"),
    )
    p.add_argument("--restore", action="store_true")
    a = p.parse_args()
    check(a.stage, a.restore)
