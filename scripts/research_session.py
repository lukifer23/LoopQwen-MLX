"""Open a new explicitly authorized execution window for native research work."""

import time
from datetime import datetime
from pathlib import Path
from loopqwen.io import save, read
from loopqwen.worker import acquire

if __name__ == "__main__":
    acquire()
    path = Path("outputs/research/active-session.json")
    now = time.time()
    state = read("research/state.json")
    prior = (
        read(path)["start"]
        if path.exists()
        else datetime.fromisoformat(
            state["session_start_utc"].replace("Z", "+00:00")
        ).timestamp()
    )
    if (
        not (path.parent / f"closed-{int(prior)}.json").exists()
        and now - prior < state["limits"]["wall_seconds"]
        and (path.exists() or (path.parent / f"budget-{int(prior)}.json").exists())
    ):
        raise SystemExit("Current local session is still active")
    save(path, dict(start=now))
    print("Opened bounded research session", int(now))
