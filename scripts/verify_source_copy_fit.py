"""Reload the fitted two-read assistant and compare full-vocabulary inference with the supervised path."""

from pathlib import Path
from time import monotonic
import mlx.core as mx
from mlx.utils import tree_flatten

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.research.conditional_features import CACHE
from loopqwen.research.source_copy_fit import extract
from loopqwen.shared_query_assistant import SharedQueryAssistant
from loopqwen.worker import acquire
from scripts.register_source_copy import FIT, RESULT
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

OUT = Path("docs/measurements/source-copy-fit-01-cold.json")


def run():
    root = RESULT / FIT
    committed(root / "checksums.json")
    cfg = read(root / "registration.json")
    if not (root / "step.safetensors").is_file():
        return
    started = monotonic()
    acquire()
    mx.reset_peak_memory()
    actor = SharedQueryAssistant(cfg["checkpoint"], root)
    net = actor._tool_model
    folder = CACHE / FIT / digest(root / "registration.json")
    idx = unpack(root / "feature-index.json.gz")
    assert all(digest(folder / r["file"]) == r["sha256"] for r in idx)
    data = {r["id"]: r for r in unpack(Path(cfg["training"]))}
    parent = mx.load(str(RESULT / cfg["parent"] / "step.safetensors"))
    actual = dict(tree_flatten(net.step.parameters()))
    assert all(mx.array_equal(v, actual[k]).item() for k, v in parent.items())
    probes = set(cfg["probes"])
    checks = []
    for group in ("conversation_read", "teacher_action", "calculate", "quantity"):
        r = min(
            (r for r in idx if r["id"] in probes and r["group"] == group),
            key=lambda r: fingerprint([28731, r["id"]]),
        )
        w = data[r["id"]]
        fresh = extract(net, w, True)
        saved = mx.load(str(folder / r["file"]))
        assert fresh.keys() == saved.keys() and all(
            mx.array_equal(a, saved[k]).item() for k, a in fresh.items()
        )
        for j in sorted({0, len(w["y"]) // 2, len(w["y"]) - 1}):
            ids = w["p"] + w["y"][:j]
            cache = net.make_cache()
            for first in range(0, len(ids) - 1, 128):
                net.populate_cache(
                    mx.array([ids[first : min(first + 128, len(ids) - 1)]]), cache
                )
            h = net.last_hidden(mx.array([[ids[-1]]]), cache)
            logp = net.project_cached(h, cache)[0, 0]
            label = w["y"][j]
            training, _, _ = net.step.pointer.selected(
                saved["two"][:, j : j + 1],
                saved["values"],
                saved["ids"],
                r["offset"] + j,
                mx.array([[label]]),
                net.project,
            )
            exact = net.step.pointer.full(
                saved["two"][:, j : j + 1],
                saved["values"],
                saved["ids"],
                r["offset"] + j,
                net.project,
            )[0, 0]
            checks.append(
                dict(
                    id=r["id"],
                    target=j,
                    selected_error=abs(exact[label].item() - training.item()),
                    cached_error=abs(logp[label].item() - training.item()),
                    sum_error=abs(mx.exp(logp).sum().item() - 1),
                    top=mx.argmax(exact).item(),
                    cached_top=mx.argmax(logp).item(),
                )
            )
    verified = all(
        r["selected_error"] < 1e-4
        and r["sum_error"] < 1e-4
        and r["cached_error"] < 0.25
        for r in checks
    )
    persist(
        OUT,
        dict(
            passed=verified,
            fit=digest(root / "checksums.json"),
            frozen_tensors=len(parent),
            source_files=len(idx),
            identical_fresh_features=4,
            checks=checks,
            seconds=monotonic() - started,
            peak_bytes=mx.get_peak_memory(),
        ),
    )
    print("Source mixture cold replay", verified, flush=True)


if __name__ == "__main__":
    run()
