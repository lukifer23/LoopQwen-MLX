"""Open a new explicitly authorized execution window for native research work."""

import time
from pathlib import Path
from loopqwen.io import save, read
from loopqwen.worker import acquire

if __name__ == "__main__":
    acquire()
    path = Path("outputs/research/active-session.json")
    now = time.time()
    if (
        path.exists()
        and now - read(path)["start"]
        < read("research/state.json")["limits"]["wall_seconds"]
    ):
        raise SystemExit("Current local session is still active")
    save(path, dict(start=now))
    print("Opened bounded research session", int(now))
