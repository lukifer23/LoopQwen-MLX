"""Cooperative project-local GPU lease. The kernel drops it when the worker exits."""

import fcntl
import os
import sys
from pathlib import Path

_lease = None
CACHE_BYTES = 256 * 1024**2


def acquire():
    global _lease
    if _lease is not None:
        return
    root = Path(__file__).resolve().parent.parent
    if not (root / "pyproject.toml").is_file():
        root = Path.cwd()
    path = root / "outputs" / "gpu-worker.lock"
    path.parent.mkdir(exist_ok=True, parents=True)
    file = path.open("a+")
    try:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        file.seek(0)
        owner = file.read().strip()
        file.close()
        raise RuntimeError(
            f"Another LoopQwen GPU worker holds the project lease ({owner})."
        ) from None
    file.seek(0)
    file.truncate()
    file.write(f"pid={os.getpid()} command={Path(sys.argv[0]).name}")
    file.flush()
    _lease = file
    import mlx.core as mx

    mx.set_cache_limit(CACHE_BYTES)


def resources():
    import subprocess

    import psutil

    from .hardware import gpu_utilization

    root = Path(__file__).resolve().parent.parent
    path = root / "outputs/gpu-worker.lock"
    owner = None
    if path.is_file():
        with path.open() as f:
            try:
                fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                owner = f.read().strip()
    workers = []
    for proc in psutil.process_iter(["pid", "ppid", "name", "status"]):
        try:
            if proc.pid == os.getpid() or not proc.name().lower().startswith(
                ("python", "zsh", "bash", "sh")
            ):
                continue
            args = proc.cmdline()[1:]
            if not any(
                arg.startswith(
                    (
                        "loopqwen",
                        "scripts/phase2_",
                        "scripts/run_phase2",
                        "scripts/phase2_finish",
                    )
                )
                for arg in args
            ):
                continue
            if not Path(proc.cwd()).is_relative_to(root):
                continue
            workers.append(
                dict(
                    pid=proc.pid,
                    parent=proc.ppid(),
                    status=proc.status(),
                    adopted_by_launchd=proc.ppid() == 1,
                    command=(
                        next(
                            (a for a in args if a.endswith(".sh") and len(a) < 128),
                            "project shell",
                        )
                        if "-c" in args
                        else " ".join(args[:5])[:160]
                    ),
                    rss_mib=round(proc.memory_info().rss / 2**20, 1),
                )
            )
        except (psutil.AccessDenied, psutil.ZombieProcess, psutil.NoSuchProcess):
            continue
    pressure = subprocess.run(
        ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
        capture_output=True,
        text=True,
    )
    return dict(
        gpu_lease=owner,
        processes=workers,
        available_gib=round(psutil.virtual_memory().available / 2**30, 2),
        swap_used_gib=round(psutil.swap_memory().used / 2**30, 2),
        gpu_device_percent=gpu_utilization(),
        memory_pressure={"1": "normal", "2": "warning", "4": "critical"}.get(
            pressure.stdout.strip(), "unknown"
        ),
    )
