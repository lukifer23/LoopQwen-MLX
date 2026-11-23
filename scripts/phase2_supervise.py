"""One study conductor per checkout. Forward cancellation to its entire process group."""

import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def supervise(argv, lock_path):
    lock_path.parent.mkdir(exist_ok=True, parents=True)
    with lock_path.open("a+") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            file.seek(0)
            raise SystemExit(f"Study already running under PID {file.read().strip()}")
        file.seek(0)
        file.truncate()
        file.write(str(os.getpid()))
        file.flush()
        proc = subprocess.Popen(
            argv,
            start_new_session=True,
            pass_fds=(file.fileno(),),
            env={
                **os.environ,
                "LOOPQWEN_STUDY_FD": str(file.fileno()),
                "LOOPQWEN_STUDY_LOCK": str(lock_path.resolve()),
            },
        )
        canceled = []

        def forward(sig, _frame):
            canceled.append(sig)
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        handlers = {
            s: signal.signal(s, forward)
            for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
        }
        try:
            result = proc.wait()
            if canceled or result:
                # A shell can exit before its trainer finishes the current atomic update.
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    return result
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    try:
                        os.killpg(proc.pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.1)
                else:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            return result
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--check"]:
        lock_path = Path(sys.argv[2]).resolve()
        fd = os.environ.get("LOOPQWEN_STUDY_FD")
        if (
            fd is None
            or not lock_path.is_file()
            or not os.path.samestat(os.fstat(int(fd)), lock_path.stat())
        ):
            raise SystemExit("Start through the study supervisor")
        fcntl.flock(int(fd), fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.exit(0)
    if sys.argv[1:2] == ["--phase3"]:
        code = supervise(
            [
                "/bin/sh",
                "-c",
                Path("scripts/run_phase3.sh").read_text(),
                "scripts/run_phase3.sh",
                "--supervised",
                *sys.argv[2:],
            ],
            Path("outputs/phase3/study.lock"),
        )
        sys.exit(128 - code if code < 0 else code)
    finish = sys.argv[1:] == ["--finish"]
    script = Path("scripts/phase2_finish.sh" if finish else "scripts/run_phase2.sh")
    # A live checkout may change. Give sh an immutable command body for this invocation.
    args = ["/bin/sh", "-c", script.read_text(), str(script), "--supervised"]
    code = supervise(
        args,
        Path("outputs/phase2/study.lock"),
    )
    sys.exit(128 - code if code < 0 else code)
