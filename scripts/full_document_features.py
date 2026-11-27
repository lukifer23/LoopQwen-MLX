"""Qualify native full-length training features for twelve questions on one original Cargo.lock."""

import argparse
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import mlx.core as mx
import psutil

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.long_training_memory import DocumentMemory
from loopqwen.research.conditional_features import CACHE
from loopqwen.research.native_function_data import token_pair
from loopqwen.shared_query_assistant import SharedQueryAssistant
from scripts.learner_prefix_material import FIT, RESULT
from scripts.long_branch_sources import CACHE as SOURCES
from scripts.long_branch_sources import OUT as MATERIAL
from scripts.long_branch_sources import SPECS
from scripts.program_language_material import answer, user
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed
from scripts.verify_long_training_memory import OUT as PARITY
from scripts.verify_long_training_memory import PLAN as SHORT_PLAN

PLAN = Path("research/diagnostics/full-document-features.json")
OUT = Path("docs/measurements/full-document-features.json")
DIR = CACHE / "full-document-features"


def register():
    for p in (MATERIAL, SHORT_PLAN, RESULT / FIT / "checksums.json"):
        committed(p)
    doc = read(MATERIAL)["sources"]["deno"]["documents"][1]
    spec = {
        "material": digest(MATERIAL),
        "numerical_plan": digest(SHORT_PLAN),
        "model": digest(RESULT / FIT / "checksums.json"),
        "implementation": digest(Path(__file__)),
        "source": doc["sha256"],
        "questions": [w["key"] for w in doc["questions"]],
        "question": "Can a single full native prefix yield independent teacher-forced states at all answer positions on twelve unrelated package lookups?",
        "protocol": "Require a passing native short branch parity check first. Enroll all twelve preselected changed crates in Deno v2.8.0 Cargo.lock, without using model predictions for selection. Use the unchanged complete source and original question, greedy reasoning-enabled template and original two-read donor. Preserve all six prefix histories and exact final KV once, and each independent tail plus all expert answer positions. Verify position/shape, source byte evidence and input identity on every branch. Re-run the first after the other eleven and require bit-identical output features. Archive every digest, processing time, MLX peak and scratch. At least one answer's deciding version must lie after token 131072. This is a training-feature prerequisite only; no answer accuracy or release inference. Stop at 660 seconds, 3 GiB MLX, 1.5 GiB available or OS memory warning.",
        "seconds": 660,
        "peak_gib": 3,
    }
    return spec, doc


def run(freeze):
    s, doc = register()
    if freeze:
        persist(PLAN, s)
        return
    committed(PLAN)
    committed(PARITY)
    assert (
        read(PLAN) == s
        and read(PARITY)["passed"]
        and read(PARITY)["plan"] == digest(SHORT_PLAN)
    )
    directory = DIR / digest(PLAN)
    directory.mkdir(exist_ok=True, parents=True)
    p = SOURCES / SPECS["deno"][3][1]
    assert digest(p) == doc["sha256"]
    raw = p.read_bytes()
    xml = raw.decode()
    filename = SPECS["deno"][1]
    first = monotonic()
    last = 0
    peak = 0

    def check(_=None):
        nonlocal last, peak
        now = monotonic()
        if now - last < 0.8:
            return
        last = now
        peak = max(peak, mx.get_peak_memory())
        if (
            now - first > s["seconds"]
            or peak > s["peak_gib"] * 2**30
            or psutil.virtual_memory().available < 1.5 * 2**30
            or subprocess.check_output(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
            ).strip()
            != "1"
        ):
            raise RuntimeError("Full-length native feature resource limit")

    mx.reset_peak_memory()
    actor = SharedQueryAssistant("exports/tool-language-packed-05", RESULT / FIT)
    tok = actor.tokenizer
    net = actor._tool_model
    xs = []
    for w in doc["questions"]:
        q = (
            f"Here is the complete {filename} file.\n\n{xml}\nEnd of file.\n\n"
            + w["question"]
        )
        p, y = token_pair(
            tok, {"messages": [user(q)], "tools": [], "target": answer(w["answer"])}
        )
        assert (
            fingerprint(p) == w["input_sha256"]
            and raw[slice(*w["source_bytes"])].decode() == w["answer"]
        )
        xs.append((w, p, y))
    length = doc["common_prompt_tokens"]
    prefix = xs[0][1][:length]
    assert fingerprint(prefix) == doc["common_ids"] and all(
        p[:length] == prefix for _, p, _ in xs
    )
    entries = []
    try:
        with TemporaryDirectory(dir=directory) as tmp:
            shared = DocumentMemory(net, prefix, tmp, check)
            opening_s = monotonic() - first
            histories = [
                slot.offset
                for slot in shared.disk.cache.slots.values()
                if hasattr(slot, "physical_length")
            ]
            assert histories == [length] * 6
            scratch = shared.disk.scratch_bytes
            for i, (w, p, y) in enumerate([*xs, xs[0]]):
                check()
                start = monotonic()
                z = shared.branch(p, y)
                a = {
                    k: z[k]
                    for k in ("pre", "ordinary", "tail_keys", "tail_values", "labels")
                }
                mx.eval(a)
                dest = directory / (str(i) + ".safetensors")
                mx.save_safetensors(str(dest), a)
                if i == 0:
                    kv = mx.load(str(z["prefix_file"]))
                    assert kv["k"].shape == kv["v"].shape and kv["k"].shape[2] == length
                    shutil.copyfile(
                        z["prefix_file"], directory / "document.safetensors"
                    )
                    assert digest(z["prefix_file"]) == digest(
                        directory / "document.safetensors"
                    )
                    del kv
                rec = {
                    "key": w["key"],
                    "opening": len(p),
                    "input_sha256": fingerprint(p),
                    "target_sha256": fingerprint(y),
                    "positions": len(y),
                    "offset": z["offset"],
                    "source": w["source_tokens"],
                    "tail": z["tail_keys"].shape[2],
                    "sha256": digest(dest),
                    "bytes": dest.stat().st_size,
                    "seconds": monotonic() - start,
                }
                assert rec["tail"] + length == len(p) + len(y) - 1
                if i == len(xs):
                    assert rec["sha256"] == entries[0]["sha256"]
                    dest.unlink()
                else:
                    entries.append(rec)
            check()
        assert len(entries) == 12 and any(r["source"][0] > 131072 for r in entries)
        out = {
            "passed": True,
            "rows": entries,
            "histories": histories,
            "repeated": entries[0]["sha256"],
            "prefix_tokens": length,
            "document_kv": digest(directory / "document.safetensors"),
            "document_bytes": (directory / "document.safetensors").stat().st_size,
            "staging_s": opening_s,
            "scratch": scratch,
        }
    except RuntimeError as e:
        if str(e) != "Full-length native feature resource limit":
            raise
        out = {"passed": False, "reason": str(e), "completed": entries}
    persist(
        OUT,
        dict(
            **out,
            plan=digest(PLAN),
            seconds=monotonic() - first,
            peak_bytes=max(peak, mx.get_peak_memory()),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("register", "measure"))
    run(p.parse_args().action == "register")
