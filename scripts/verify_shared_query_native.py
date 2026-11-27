"""Independently regenerate final-layer training memory and replay learned conclusions on Metal."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.conditional_features import feature_root
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from loopqwen.research.registered import registered
from loopqwen.research.shared_query_features import FEATURES, FIT, OLD, RESULTS, arrays
from loopqwen.research.shared_query_fit import probe
from loopqwen.shared_query_assistant import SharedQueryAssistant, from_backbone
from loopqwen.worker import acquire


def check(name, restore):
    acquire()
    s = registered(name)
    root = RESULTS / name
    fs = registered(FEATURES)
    features = RESULTS / FEATURES
    assert read(root / "decision.json")["passed"]
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
        for g in ("strategy", "report_action", "full_rows")
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
    if name == FIT:
        model = SharedQueryAssistant(s["checkpoint"], root)._tool_model
        # This process has no reference to the optimizer or training graph.
        native = probe(model, idx, folder, unpack(Path(s["evidence"])), s)
        saved = read(root / "probes.json")["after"]
        assert [r["id"] for r in native] == [r["id"] for r in saved]
        errors = {
            m: max(abs(r[m] - a[m]) for r, a in zip(native, saved, strict=True))
            for m in ("ce", "ordinary_ce", "kl", "citation_nll")
        }
        assert max(errors.values()) <= 1e-5
        report.update(
            fit=digest(root / "checksums.json"),
            trained_positions=sum(r["positions"] for r in native),
            max_errors=errors,
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
    p.add_argument("stage", choices=(FEATURES, FIT))
    p.add_argument("--restore", action="store_true")
    a = p.parse_args()
    check(a.stage, a.restore)
