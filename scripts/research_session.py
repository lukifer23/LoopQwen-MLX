"""Open or archive a bounded native research window."""

import argparse
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from loopqwen.io import digest, read, save
from loopqwen.worker import acquire


def close(path, started, state, now, *, checkpoint=None):
    closed = path.parent / f"closed-{int(started)}.json"
    if closed.exists():
        raise SystemExit("This execution window is already closed")
    ledger = path.parent / f"budget-{int(started)}.json"
    jobs = read(ledger) if ledger.exists() else []
    if any(r.get("status") == "unclosed reservation" for r in jobs):
        raise SystemExit("An unresolved native reservation remains")
    registry = read("research/registry.json")
    interim = None
    if checkpoint is not None and {r["job"] for r in jobs} != {checkpoint}:
        raise SystemExit("The checkpoint must cover the entire window")
    for name in dict.fromkeys(r["job"] for r in jobs):
        if checkpoint == name:
            from scripts.seal_staged_window import check

            interim = check(name, int(started))
            continue
        src = Path("research/results") / name
        if (
            not (src / "checksums.json").exists()
            or digest(src / "checksums.json")
            != registry["experiments"][name]["archive"]
        ):
            raise SystemExit("Archive the completed job first: " + name)
        if not any(r["id"] == name for r in state["completed"]):
            state["completed"].append(
                dict(
                    id=name,
                    promoted=read(src / "decision.json")["passed"],
                    scope=read(src / "registration.json")["question"],
                    decision=str(src / "decision.json"),
                )
            )
    record = dict(
        start=started,
        closed=now,
        wall_seconds=now - started,
        native_seconds=sum(r["seconds"] for r in jobs),
        jobs=jobs,
    )
    if interim is not None:
        record["checkpoint"] = interim
    state.update(active=None, closed_session_start=started)
    state["native_window"].update(
        status="closed", end_utc=datetime.fromtimestamp(now, timezone.utc).isoformat()
    )
    # Publish the evidence ledger before closing the local lease window.
    save(f"research/sessions/window-{int(started)}.json", record)
    save(closed, record)
    save("research/state.json", state)
    print(
        "Archived bounded research session",
        int(started),
        record["native_seconds"],
        "native seconds",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--close",
        action="store_true",
        help="Record completed jobs and close the current execution window",
    )
    parser.add_argument(
        "--checkpoint",
        choices=["staged-query-transfer-01"],
        help="Close the intermediate window against the committed complete arm archive",
    )
    args = parser.parse_args()
    if args.checkpoint and not args.close:
        parser.error("--checkpoint requires --close")
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
    if args.close:
        close(path, prior, state, now, checkpoint=args.checkpoint)
        return
    if (
        not (path.parent / f"closed-{int(prior)}.json").exists()
        and now - prior < state["limits"]["wall_seconds"]
        and (path.exists() or (path.parent / f"budget-{int(prior)}.json").exists())
    ):
        raise SystemExit("Current local session is still active")
    stamp = datetime.fromtimestamp(now, timezone.utc).isoformat()
    state.update(
        session_start_utc=stamp,
        starting_revision=subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip(),
        native_window=dict(start_utc=stamp, status="open"),
    )
    save(path, dict(start=now))
    save("research/state.json", state)
    print("Opened bounded research session", int(now))


if __name__ == "__main__":
    main()
