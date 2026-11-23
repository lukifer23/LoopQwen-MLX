import hashlib
import json
from pathlib import Path


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def checkpoint_location(root):
    path = Path(root)
    if (path / "state.json").exists():
        return path
    if not (path / "latest.json").exists():
        raise FileNotFoundError(
            f"No saved checkpoint in {root}. Run the corresponding train config first."
        )
    return path / read(path / "latest.json")["directory"]


def evaluation_record(directory, arm):
    """Locate revision-bound results with support for historical output layouts."""
    pointer = directory / (arm + ".revision.json")
    return directory / (read(pointer)["path"] if pointer.is_file() else arm + ".json")
