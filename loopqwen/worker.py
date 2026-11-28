"""Cooperative user-wide GPU lease. The kernel drops it when the worker exits."""

import fcntl
import os
import sys
from pathlib import Path

_lease = None
_stable_lease = None
CACHE_BYTES = 256 * 1024**2


def lease_path():
    return Path.home() / "Library" / "Caches" / "LoopQwen" / "gpu-worker.lock"


def stable_lease_path():
    return (
        Path.home() / "Library" / "Application Support" / "LoopQwen" / "gpu-worker.lock"
    )


def child_lease():
    acquire()
    legacy, stable = _lease.fileno(), _stable_lease.fileno()
    return dict(
        pass_fds=(legacy, stable),
        env={
            **os.environ,
            "LOOPQWEN_LEASE_FD": str(legacy),
            "LOOPQWEN_STABLE_LEASE_FD": str(stable),
        },
    )


def check_study(root):
    import psutil

    ancestors = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    for phase in ("phase2", "phase3", "phase4", "phase5", "phase6", "phase7", "rsi1"):
        path = root / "outputs" / phase / "study.lock"
        if not path.is_file():
            continue
        with path.open() as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                owner = lock.read().strip()
                if not owner.isdigit() or int(owner) not in ancestors:
                    raise RuntimeError(f"{phase} conductor {owner} owns this checkout")


def _lock(path, inherited):
    path.parent.mkdir(mode=0o700, exist_ok=True, parents=True)
    if inherited is not None:
        fd = int(inherited)
        if not os.path.samestat(os.fstat(fd), path.stat()):
            raise RuntimeError("Inherited descriptor does not identify the GPU lease")
        file = os.fdopen(fd, "r+")
    else:
        file = path.open("a+")
    try:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        file.seek(0)
        owner = file.read().strip()
        file.close()
        raise RuntimeError(
            f"Another LoopQwen GPU worker holds the user-wide GPU lease ({owner})."
        ) from None
    if inherited is None:
        file.seek(0)
        file.truncate()
        file.write(
            f"pid={os.getpid()} checkout={Path.cwd()} command={Path(sys.argv[0]).name}"
        )
        file.flush()
    return file


def acquire():
    global _lease, _stable_lease
    if _lease is not None:
        return
    check_study(Path(__file__).resolve().parents[1])
    # Cache cleaners can unlink a locked inode. The persistent lock keeps new
    # workers exclusive. Also take the historical lock for installed releases.
    stable = _lock(
        stable_lease_path(), os.environ.pop("LOOPQWEN_STABLE_LEASE_FD", None)
    )
    try:
        _lease = _lock(lease_path(), os.environ.pop("LOOPQWEN_LEASE_FD", None))
    except BaseException:
        stable.close()
        raise
    _stable_lease = stable
    import mlx.core as mx

    mx.set_cache_limit(CACHE_BYTES)


def resources():
    import subprocess

    import psutil

    from .hardware import gpu_utilization

    root = Path(__file__).resolve().parent.parent
    owner = None
    for path in (stable_lease_path(), lease_path()):
        try:
            file = path.open()
        except FileNotFoundError:
            continue
        with file as f:
            try:
                fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                owner = f.read().strip()
                break
    checkout_owns_lease = owner is not None and f" checkout={root} command=" in owner
    lease_files = {str(stable_lease_path()), str(lease_path())}
    workers = []
    for proc in psutil.process_iter(["pid", "ppid", "name", "status"]):
        try:
            if proc.pid == os.getpid() or not proc.name().lower().startswith(
                ("python", "zsh", "bash", "sh")
            ):
                continue
            args = proc.cmdline()[1:]
            # Cold replay runs an installed wheel with -I outside the checkout.
            # Its inherited lease descriptor still identifies the real worker,
            # even if the original supervising process has already exited.
            lease_open = checkout_owns_lease and any(
                f.path in lease_files for f in proc.open_files()
            )
            named_worker = any(
                arg.startswith(
                    (
                        "loopqwen",
                        "scripts.middle_depth_",
                        "scripts/middle_depth_",
                        "scripts/run_phase3",
                        "scripts/phase3_",
                        "scripts/phase4",
                        "scripts/run_phase5",
                        "scripts/run_phase6",
                        "scripts/phase2_",
                        "scripts/run_phase2",
                        "scripts/phase2_finish",
                    )
                )
                for arg in args
            )
            if not lease_open and not (
                named_worker and Path(proc.cwd()).is_relative_to(root)
            ):
                continue
            if lease_open and "-I" in args and "-c" in args:
                command = "isolated Python with inherited GPU lease"
            elif "-c" in args:
                command = next(
                    (a for a in args if a.endswith(".sh") and len(a) < 128),
                    "project shell",
                )
            else:
                command = " ".join(args[:5])[:160]
            workers.append(
                dict(
                    pid=proc.pid,
                    parent=proc.ppid(),
                    status=proc.status(),
                    adopted_by_launchd=proc.ppid() == 1,
                    checkout_lease_open=lease_open,
                    command=command,
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
        gpu_lease_scope="user",
        checkout=str(root),
        processes=workers,
        available_gib=round(psutil.virtual_memory().available / 2**30, 2),
        swap_used_gib=round(psutil.swap_memory().used / 2**30, 2),
        gpu_device_percent=gpu_utilization(),
        memory_pressure={"1": "normal", "2": "warning", "4": "critical"}.get(
            pressure.stdout.strip(), "unknown"
        ),
    )
