"""Read a frozen study only when its registration and material still match the registry."""

from pathlib import Path

from ..io import digest, read


def registered(name):
    registration = Path("research/experiments") / (name + ".json")
    registry = read("research/registry.json")
    if digest(registration) != registry["experiments"][name]["registration"]:
        raise ValueError("Registered study changed: " + name)
    spec = read(registration)
    for k in ("material", "training"):
        if (
            k in spec
            and isinstance(spec[k], str)
            and spec[k].startswith("research/data/")
        ):
            path = Path(spec[k])
            if digest(path) != registry["datasets"][str(path)]["sha256"]:
                raise ValueError("Frozen material changed: " + str(path))
    return spec
