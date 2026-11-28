"""Exclusive, cumulative wall budgets and preflight checks for the native study."""

import ctypes
import os
import shutil
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import psutil

from loopqwen.hardware import require_metal
from loopqwen.io import read, save
from loopqwen.worker import acquire

ROOT = Path("docs/measurements/middle-depth")
BASE = Path("exports/tool-language-packed-05")
BUDGET = {"profile": 3600, "fit": 18000, "short": 7200, "long": 14400}
GIB = 2**30


class DevelopmentLimit(InterruptedError):
    def __init__(self, current, remaining):
        self.snapshot = current
        self.remaining_seconds = remaining
        super().__init__(
            f"Native development limit: {current}; remaining worker seconds {remaining:.2f}"
        )


class ProcessUsage(ctypes.Structure):
    # Darwin rusage_info_v2, sys/resource.h. The 16 counters between UUID and
    # diskio are CPU, memory and child-process statistics in the published ABI.
    _fields_ = [
        ("uuid", ctypes.c_uint8 * 16),
        ("resource_counters", ctypes.c_uint64 * 16),
        ("bytes_read", ctypes.c_uint64),
        ("bytes_written", ctypes.c_uint64),
    ]


def disk_io():
    usage = ProcessUsage()
    proc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    if proc.proc_pid_rusage(os.getpid(), 2, ctypes.byref(usage)) != 0:
        raise OSError(ctypes.get_errno(), "proc_pid_rusage")
    return dict(read=usage.bytes_read, written=usage.bytes_written)


def sample():
    return dict(
        pressure=int(
            subprocess.check_output(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
            )
        ),
        available=psutil.virtual_memory().available,
        swap=psutil.swap_memory().used,
        free=shutil.disk_usage(".").free,
        active=mx.get_active_memory(),
        peak=mx.get_peak_memory(),
    )


def recover_charge(path, ended=None):
    """Settle a lost process after the exclusive lease is regained.

    The conductor can supply the observed exit time. After machine loss the
    first recovery time is a conservative upper bound, labeled explicitly.
    """
    r = read(path)
    if r["status"] != "running":
        return r
    if psutil.pid_exists(r["pid"]):
        proc = psutil.Process(r["pid"])
        if (
            proc.status() != psutil.STATUS_ZOMBIE
            and abs(proc.create_time() - r.get("process_start", r["dispatch"])) < 20
        ):
            raise RuntimeError("A recorded native process is still live")
    real_end = time.time() if ended is None else ended
    r.update(
        status="interrupted",
        seconds=max(r.get("observed_seconds", 0), real_end - r["dispatch"]),
        reason="Native process exited before its final receipt",
        final=r.get("last", r["initial"]),
        cost_basis="upper bound to next lease acquisition"
        if ended is None
        else "observed process exit",
    )
    save(path, r)
    return r


@contextmanager
def worker(stage, job):
    require_metal()
    acquire()
    ledger = ROOT / "charges"
    ledger.mkdir(exist_ok=True, parents=True)
    past = [recover_charge(p) for p in ledger.glob("*.json")]
    credit = BUDGET[stage] - sum(r["seconds"] for r in past if r["stage"] == stage)
    if job.startswith("local-workflows/"):
        used = sum(
            r["seconds"] for r in past if r["job"].startswith("local-workflows/")
        )
        credit = min(credit, 360 - used)
    if job.startswith("specialist-transfer/"):
        used = sum(
            r["seconds"] for r in past if r["job"].startswith("specialist-transfer/")
        )
        credit = min(credit, 300 - used)
    if "/short/depth/" in job:
        used = sum(r["seconds"] for r in past if "/short/depth/" in r["job"])
        credit = min(credit, 420 - used)
    if stage == "long" and job.endswith("/recall-depth"):
        used = sum(r["seconds"] for r in past if r["job"].endswith("/recall-depth"))
        credit = min(credit, 480 - used)
    if stage == "long" and job.endswith("/manual-recall"):
        from scripts.middle_depth_manuals import LIMIT

        used = sum(r["seconds"] for r in past if r["job"].endswith("/manual-recall"))
        credit = min(credit, LIMIT - used)
    if job.startswith("standard-five/"):
        used = sum(r["seconds"] for r in past if r["job"].startswith("standard-five/"))
        credit = min(credit, 900 - used)
    if job.startswith("installed-benchmark/"):
        used = sum(
            r["seconds"] for r in past if r["job"].startswith("installed-benchmark/")
        )
        credit = min(credit, 900 - used)
    if job.startswith("standalone-pair/"):
        spent = sum(
            r["seconds"] for r in past if r["job"].startswith("standalone-pair/")
        )
        credit = min(credit, 240 - spent)
    if job.startswith("quantized/") or "/short/quantized/" in job:
        used = sum(
            r["seconds"]
            for r in past
            if r["stage"] == stage
            and (r["job"].startswith("quantized/") or "/short/quantized/" in r["job"])
        )
        credit = min(credit, 300 - used)
    if "/short/epoch-reading/" in job:
        used = sum(r["seconds"] for r in past if "/short/epoch-reading/" in r["job"])
        credit = min(credit, 600 - used)
    if job == "staged-visit-parity":
        credit = min(credit, 180 - sum(r["seconds"] for r in past if r["job"] == job))
    if stage == "long" and job.endswith("/epoch-one"):
        prior_book = sum(
            r["seconds"]
            for r in past
            if r["stage"] == "long" and r["job"].endswith("/epoch-one")
        )
        credit = min(credit, 1200 - prior_book)
    path = ledger / (uuid.uuid4().hex + ".json")
    limits = {"cache_bytes": 64 * 2**20}
    recovery = read(Path("research/diagnostics/middle-depth/allocator-recovery.json"))
    if stage in recovery["applies_to"]:
        limits = {k: recovery[k] for k in ("cache_bytes", "memory_guideline_bytes")}
        mx.set_memory_limit(limits["memory_guideline_bytes"])
    mx.set_cache_limit(limits["cache_bytes"])
    mx.reset_peak_memory()
    initial = sample()
    start = time.monotonic()
    record = dict(
        job=job,
        allocator=limits,
        stage=stage,
        dispatch=time.time(),
        pid=os.getpid(),
        process_start=psutil.Process().create_time(),
        initial_process_disk_io=disk_io(),
        initial=initial,
        status="running",
        seconds=0,
    )
    save(path, record)
    last = 0.0
    heartbeat = start

    def guard(*_, force=False):
        nonlocal last, heartbeat
        now = time.monotonic()
        if not force and now - last < 0.5:
            return
        last = now
        current = sample()
        record["last"] = current
        if now - heartbeat >= 5:
            record["observed_seconds"] = now - start
            save(path, record)
            heartbeat = now
        if (
            now - start >= credit
            or current["pressure"] != 1
            or current["available"] < 1.5 * GIB
            or current["swap"] - initial["swap"] > GIB
            or current["peak"] >= 8 * GIB
            or current["free"] < 24 * GIB
        ):
            raise DevelopmentLimit(current, credit - (now - start))

    try:
        guard(force=True)
        yield guard
        guard(force=True)
        record["status"] = "finished"
    except BaseException as error:
        record.update(status="stopped", reason=repr(error))
        raise
    finally:
        record.update(seconds=time.monotonic() - start, final=sample())
        record["final_process_disk_io"] = disk_io()
        save(path, record)
