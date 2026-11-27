"""Exclusive, cumulative wall budgets and preflight checks for the native study."""
from contextlib import contextmanager
from pathlib import Path
import os
import shutil
import subprocess
import time
import uuid

import mlx.core as mx
import psutil

from loopqwen.hardware import require_metal
from loopqwen.io import read, save
from loopqwen.worker import acquire

ROOT = Path("docs/measurements/middle-depth")
BASE = Path("exports/tool-language-packed-05")
BUDGET = {"profile": 3600, "fit": 18000, "short": 7200, "long": 14400}
GIB = 2**30


def sample():
    return dict(pressure=int(subprocess.check_output(
        ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True)),
        available=psutil.virtual_memory().available, swap=psutil.swap_memory().used,
        free=shutil.disk_usage(".").free, active=mx.get_active_memory(),
        peak=mx.get_peak_memory())


@contextmanager
def worker(stage, job):
    require_metal()
    acquire()
    ledger = ROOT / "charges"
    ledger.mkdir(exist_ok=True, parents=True)
    past = [read(p) for p in ledger.glob("*.json")]
    if any(r["status"] == "running" for r in past):
        raise RuntimeError("Reconcile interrupted worker charge first")
    credit = BUDGET[stage] - sum(r["seconds"] for r in past if r["stage"] == stage)
    path = ledger / (uuid.uuid4().hex + ".json")
    mx.set_cache_limit(64 * 2**20)
    mx.reset_peak_memory()
    initial = sample()
    start = time.monotonic()
    record = dict(job=job, stage=stage, dispatch=time.time(), pid=os.getpid(),
                  initial=initial, status="running", seconds=0)
    save(path, record)
    last = 0.
    def guard(*_, force=False):
        nonlocal last
        now = time.monotonic()
        if not force and now - last < 0.5:
            return
        last = now
        current = sample()
        record["last"] = current
        if (now - start >= credit or current["pressure"] != 1 or
            current["available"] < 1.5 * GIB or current["swap"] - initial["swap"] > GIB
            or current["peak"] >= 8 * GIB or current["free"] < 24 * GIB):
            raise InterruptedError(f"Native development limit: {current}")
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
        save(path, record)
