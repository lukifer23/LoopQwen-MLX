"""Record bounded native-process restarts for studies that explicitly allow them."""

from ..io import read, save


def relinquish(root, spec, error, progress):
    if (
        not spec.get("max_pressure_restarts")
        or not isinstance(error, MemoryError)
        or not str(error).startswith("System memory pressure")
    ):
        return False
    path = root / "pressure-stops.json"
    stops = read(path) if path.exists() else []
    stops.append({"reason": str(error), **progress})
    save(path, stops)
    print("Releasing native process", spec["id"], progress, flush=True)
    return len(stops) <= spec["max_pressure_restarts"]


def budget(root, spec):
    from .core import Budget

    try:
        return Budget(spec["id"], spec["limits"])
    except (MemoryError, TimeoutError) as error:
        if not spec.get("max_pressure_restarts"):
            raise
        if not relinquish(root, spec, error, {"phase": "before model allocation"}):
            save(
                root / "decision.json",
                {
                    "passed": False,
                    "status": "interrupted",
                    "reason": str(error),
                    "durable": 0,
                },
            )
        return None
