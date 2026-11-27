"""Run the frozen two-model changed-dependency comparison on complete release files."""

import argparse
import gc
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import mlx.core as mx
import psutil

from loopqwen.evaluation.mechanism import packed
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.shared_query_assistant import SharedQueryAssistant
from loopqwen.tool_compare import DenseAssistant
from scripts import full_document_adjoint as adjoint
from scripts import verify_document_generation as parity
from scripts.full_document_features import DIR
from scripts.learner_prefix_material import FIT, RESULT
from scripts.long_branch_sources import CACHE
from scripts.long_dependency_sources import END, OUT as PANEL, PREFIX
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

PLAN = Path("research/diagnostics/full-document-relational-pair.json")
OUTPUT = Path("docs/measurements/full-document-relations")
EXPORT = "exports/tool-language-packed-05"


class LimitReached(RuntimeError):
    pass


def install(folder, receipt):
    weights = DIR / digest(adjoint.PLAN) / "step.safetensors"
    assert digest(weights) == receipt["weights"]
    spec = read(RESULT / FIT / "registration.json")
    spec.update(content_read=True, full_document_plan=digest(adjoint.PLAN))
    persist(folder / "registration.json", spec)
    shutil.copyfile(weights, folder / "step.safetensors")
    persist(
        folder / "fit.json",
        dict(
            weights=receipt["weights"],
            registration=digest(folder / "registration.json"),
            backbone=read(RESULT / FIT / "fit.json")["backbone"],
        ),
    )
    return SharedQueryAssistant(EXPORT, folder)


def run():
    for p in (PLAN, PANEL, adjoint.PLAN, parity.PLAN):
        committed(p)
    spec = read(PLAN)
    assert digest(PANEL) == spec["material"]
    assert digest(adjoint.PLAN) == spec["training_prerequisite"]
    assert digest(parity.PLAN) == spec["public_parity_prerequisite"]
    target = OUTPUT / "collection.json"
    assert not target.exists()
    evidence = {}
    for name, p, plan in (
        ("fit", adjoint.OUT, adjoint.PLAN),
        ("parity", parity.OUT, parity.PLAN),
    ):
        if p.is_file():
            committed(p)
            data = read(p)
            assert data["plan"] == digest(plan)
            evidence[name] = dict(passed=data["passed"], sha256=digest(p))
    if any(not r["passed"] for r in evidence.values()):
        persist(
            target,
            dict(
                plan=digest(PLAN),
                complete=False,
                closed="failed prerequisite",
                evidence=evidence,
            ),
        )
        return
    assert len(evidence) == 2, "Training and public generation receipts must be sealed"
    panel = read(PANEL)["projects"]
    order = [
        [repo, doc["sha256"], c["input_sha256"]]
        for repo, project in panel.items()
        for doc in project["documents"]
        for c in doc["cases"]
    ]
    assert order == spec["ordered"]
    limits = spec["budget"]
    begin = monotonic()
    arms = {}
    for arm in ("candidate", "control"):
        start = monotonic()
        next_check = 0
        scratch = None
        peak_scratch = 0
        documents = []
        mx.reset_peak_memory()

        def guard(_=None, *, force=False):
            nonlocal next_check, peak_scratch
            now = monotonic()
            if not force and now < next_check:
                return
            next_check = now + 1
            if scratch is not None:
                peak_scratch = max(
                    peak_scratch, sum(p.stat().st_size for p in scratch.iterdir())
                )
            if (
                now - begin > limits["wall_seconds"]
                or now - start > limits["per_arm_seconds"]
                or peak_scratch > limits["scratch_gib_per_doc"] * 2**30
                or mx.get_peak_memory() > limits["peak_gib"] * 2**30
                or psutil.virtual_memory().available < limits["available_gib"] * 2**30
                or subprocess.check_output(
                    ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
                ).strip()
                != "1"
            ):
                raise LimitReached("registered time, scratch or memory limit")

        closed = None
        try:
            guard(force=True)
            with TemporaryDirectory() as policy:
                if arm == "candidate":
                    actor = install(Path(policy), read(adjoint.OUT))
                else:
                    actor = DenseAssistant(True, "qwen", EXPORT)
                    actor.native_plain = True
                    assert actor.language.base_source == "certified_release_text"
                load = monotonic() - start
                for repo, project in panel.items():
                    for doc in project["documents"]:
                        stage_at = monotonic()
                        file = CACHE / doc["file"]
                        assert digest(file) == doc["sha256"]
                        body = file.read_bytes().decode()
                        qs = [PREFIX + body + END + c["question"] for c in doc["cases"]]
                        with TemporaryDirectory() as directory:
                            scratch = Path(directory)
                            prefix = actor.prepare_document(
                                qs, directory, progress=guard
                            )
                            assert (
                                fingerprint(prefix.ids[: doc["common_tokens"]])
                                == doc["common_ids"]
                            )
                            assert len(prefix.ids) >= doc["common_tokens"] >= 131072
                            entry = dict(
                                project=repo,
                                document=doc["sha256"],
                                stage_s=monotonic() - stage_at,
                                stage_tokens=len(prefix.ids),
                                scratch_bytes=prefix.disk.scratch_bytes,
                                rows=[],
                            )
                            documents.append(entry)
                            peak_scratch = max(peak_scratch, prefix.disk.scratch_bytes)
                            guard(force=True)
                            for q, case in zip(qs, doc["cases"], strict=True):
                                served = actor.respond(
                                    q,
                                    max_tokens=384,
                                    max_turns=1,
                                    progress=guard,
                                    record_kv_history=True,
                                    prefill_cache_state=True,
                                    document_prefix=prefix,
                                )
                                (turn,) = served["turns"]
                                ids = turn["prompt_ids"]
                                assert (
                                    fingerprint(ids) == case["input_sha256"]
                                    and len(ids) == case["opening"]
                                )
                                assert turn["new_prompt_tokens"] + turn[
                                    "reused_prompt_tokens"
                                ] == len(ids)
                                count = len(ids) + len(turn["token_ids"]) - 1
                                assert len(turn["kv_history"]) == 6 and all(
                                    h["offset"] == h["retained"] == count
                                    for h in turn["kv_history"]
                                )
                                entry["rows"].append(served)
                                guard(force=True)
                            del prefix
                            scratch = None
                        print(
                            arm,
                            "finished complete document",
                            len(documents),
                            flush=True,
                        )
                del actor
        except LimitReached as err:
            closed = str(err)
        elapsed = monotonic() - start
        raw = OUTPUT / (arm + ".json.gz")
        packed(raw, documents)
        arms[arm] = dict(
            raw=digest(raw),
            seconds=elapsed,
            peak_bytes=mx.get_peak_memory(),
            peak_scratch=peak_scratch,
            load_s=load if documents else None,
            requests=sum(len(d["rows"]) for d in documents),
            closed=closed,
        )
        gc.collect()
        mx.clear_cache()
        if closed:
            break
    persist(
        target,
        dict(
            plan=digest(PLAN),
            evidence=evidence,
            arms=arms,
            seconds=monotonic() - begin,
            complete=len(arms) == 2
            and all(v["closed"] is None and v["requests"] == 16 for v in arms.values()),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()
