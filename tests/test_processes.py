"""The study launcher must reject duplicate conductors and clean up after cancellation."""

import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil


def test_study_process_group(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    program = (
        "import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
        "from phase2_supervise import supervise;sys.exit(supervise(sys.argv[3:],Path(sys.argv[2])))"
    )
    child = (
        "import os,sys,time;from pathlib import Path;"
        "Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(15)"
    )
    pidfile = tmp_path / "worker.pid"
    args = [
        sys.executable,
        "-c",
        program,
        str(scripts),
        str(tmp_path / "study.lock"),
        sys.executable,
        "-c",
        child,
        str(pidfile),
    ]
    leader = subprocess.Popen(args)
    try:
        for _ in range(100):
            if pidfile.is_file() and pidfile.stat().st_size:
                break
            time.sleep(0.02)
        worker = int(pidfile.read_text())
        duplicate = subprocess.run(args, capture_output=True, text=True, timeout=4)
        assert duplicate.returncode != 0 and "already running" in duplicate.stderr
        leader.send_signal(signal.SIGTERM)
        leader.wait(timeout=6)
        assert not psutil.pid_exists(worker)
    finally:
        if leader.poll() is None:
            leader.terminate()
            leader.wait(timeout=8)
