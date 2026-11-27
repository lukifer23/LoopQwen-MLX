"""Time the registered pair using lossless full-history staging and durable user requests."""

import gc
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import mlx.core as mx

from ..evaluation.mechanism import packed, unpack
from ..io import digest, read, save
from .conditional_features import CACHE
from .conditional_primary import ARMS
from .file_evidence_environment import capture
from .host_pressure import budget, relinquish
from .provider_field import FieldDatabases
from .registered import registered
from .shared_query_features import RESULTS, SOURCES
from .shared_query_transfer import measure, outcome
from .table_reentry_fit import root_for

STAGE = "staged-query-transfer-01"


def restore(root, folder, rows):
    journal = root / "request-index.json"
    index = read(journal) if journal.exists() else {a: [] for a in ARMS}
    runs = {a: [] for a in ARMS}
    for a in ARMS:
        for i, sha in enumerate(index[a]):
            f = folder / f"{a}-{i:03}.json.gz"
            assert digest(f) == sha
            r = unpack(f)
            assert r["id"] == rows[i]["id"]
            runs[a].append(r)
    return index, runs


def partial(root, runs):
    durable = {}
    for arm, records in runs.items():
        if records:
            packed(root / (arm + "-partial.json.gz"), records)
            durable[arm] = len(records)
    return durable


def expire(s, root):
    folder = CACHE / s["id"] / digest(root / "registration.json")
    _, runs = restore(root, folder, unpack(Path(s["material"]))["rows"])
    return partial(root, runs)


def run(name=STAGE):
    s = registered(name)
    assert all(digest(Path(k)) == v for k, v in s["sources"].items())
    proof = read(s["staging_receipt"])
    if not proof["passed"]:
        raise ValueError("Numerical and full-length staging qualification is required")
    root = root_for(s)
    capture(
        root,
        (
            *SOURCES,
            "loopqwen/research/shared_query_fit.py",
            "loopqwen/research/shared_query_transfer.py",
            "loopqwen/research/staged_query_transfer.py",
            "loopqwen/research/native_function_study.py",
            "loopqwen/research/provider_field.py",
            "loopqwen/tool_session.py",
            "loopqwen/tool_model.py",
            "loopqwen/offloaded_prefill.py",
            "loopqwen/reserved_cache.py",
            "loopqwen/evaluation/generation_eval.py",
            "scripts/provider_field_mutations.py",
            "scripts/repair_query_worker.py",
        ),
    )
    fit = RESULTS / s["fit_stage"]
    assert read(fit / "decision.json")["passed"]
    save(
        root / "weights.json",
        {
            "step": digest(fit / "step.safetensors"),
            "archive": digest(fit / "checksums.json"),
            "backbone": read(fit / "fit.json")["backbone"]["sha256"],
            "base": s["base_text_sha256"],
        },
    )
    folder = CACHE / name / digest(root / "registration.json")
    folder.mkdir(parents=True, exist_ok=True)
    rows = unpack(Path(s["material"]))["rows"]
    index, runs = restore(root, folder, rows)
    pending = root / "failed-attempts.json"
    failures = read(pending) if pending.exists() else []
    arm_window = root / "candidate-window.json"
    spent_before_arm = 0
    if arm_window.exists():
        ledger = Path("research/sessions") / (
            "window-" + str(read(arm_window)["window"]) + ".json"
        )
        spent_before_arm = sum(
            j["seconds"] for j in read(ledger)["jobs"] if j["job"] == name
        )
    arm_limit = min(
        s["limits"]["wall_seconds"], spent_before_arm + s["seconds_per_arm"]
    )
    guard = budget(root, {**s, "limits": {**s["limits"], "wall_seconds": arm_limit}})
    if guard is None:
        # A preallocation timeout must preserve already completed requests in the final archive.
        if (root / "decision.json").exists():
            d = read(root / "decision.json")
            d["durable"] = partial(root, runs)
            save(root / "decision.json", d)
        return
    try:
        with TemporaryDirectory() as tmp:
            fields = FieldDatabases(tmp)
            for arm in ARMS:
                if (
                    len(runs[arm]) == len(rows)
                    and (root / (arm + ".json.gz")).is_file()
                ):
                    continue
                gc.collect()
                mx.clear_cache()
                actor = None
                for i in range(len(runs[arm]), len(rows)):
                    w = rows[i]
                    start = perf_counter()
                    try:
                        guard.check()
                        actor, r = measure(s, w, arm, actor, guard, fields)
                    except (MemoryError, TimeoutError, InterruptedError) as e:
                        failures.append(
                            {
                                "arm": arm,
                                "id": w["id"],
                                "wall_s": perf_counter() - start,
                                "peak": mx.get_peak_memory(),
                                "reason": str(e),
                            }
                        )
                        save(pending, failures)
                        same = sum(
                            f["arm"] == arm and f["id"] == w["id"] for f in failures
                        )
                        if same < s["max_attempts_per_item"] and relinquish(
                            root, s, e, {"arm": arm, "index": i, "id": w["id"]}
                        ):
                            return
                        raise
                    q = r["requests"][0]
                    retries = [
                        v for v in failures if (v["arm"], v["id"]) == (arm, w["id"])
                    ]
                    q["failed_attempt_seconds"] = sum(v["wall_s"] for v in retries)
                    q["wall_s"] += q["failed_attempt_seconds"]
                    q["first_s"] += q["failed_attempt_seconds"]
                    q["peak"] = max([q["peak"], *(v["peak"] for v in retries)])
                    f = folder / f"{arm}-{i:03}.json.gz"
                    packed(f, r)
                    index[arm].append(digest(f))
                    runs[arm].append(r)
                    save(root / "request-index.json", index)
                    if (i + 1) % 8 == 0:
                        print(arm, i + 1, flush=True)
                del actor
                packed(root / (arm + ".json.gz"), runs[arm])
                if arm == "candidate":
                    save(
                        arm_window,
                        {"window": int(guard.session_start), "requests": len(rows)},
                    )
                    return
        d = outcome(rows, runs)
    except (MemoryError, TimeoutError, InterruptedError) as e:
        d = {
            "passed": False,
            "status": "interrupted",
            "reason": str(e),
            "durable": partial(root, runs),
        }
    finally:
        guard.close()
    save(root / "decision.json", d)


def verify(root, d):
    from .shared_query_transfer import verify as verify_transfer

    verify_transfer(root, d)
    index_path = root / "request-index.json"
    index = read(index_path) if index_path.exists() else {a: [] for a in ARMS}
    failures = (
        read(root / "failed-attempts.json")
        if (root / "failed-attempts.json").exists()
        else []
    )
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "individual.json.gz"
        for arm in ARMS:
            full = root / (arm + ".json.gz")
            path_for_arm = full if full.exists() else root / (arm + "-partial.json.gz")
            entries = unpack(path_for_arm) if path_for_arm.exists() else []
            for r, sha in zip(entries, index[arm], strict=True):
                packed(path, r)
                assert digest(path) == sha
                prev = [v for v in failures if (v["arm"], v["id"]) == (arm, r["id"])]
                q = r["requests"][0]
                assert q["failed_attempt_seconds"] == sum(v["wall_s"] for v in prev)
                assert all(q["peak"] >= v["peak"] for v in prev)
