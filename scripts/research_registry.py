"""Index immutable local research artifacts and check map disjointness without MLX."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def inventory():
    root = Path("research")
    historical = root / "data/historical-maps.txt.gz"
    require_sha = json.loads((root / "data/historical.json").read_text())["sha256"]
    if digest(historical) != require_sha:
        raise ValueError("Historical exclusions changed")
    seen = set(gzip.decompress(historical.read_bytes()).decode().splitlines())
    datasets = {}
    for p in sorted((root / "data").glob("*.json.gz")):
        splits = json.loads(gzip.decompress(p.read_bytes()))["splits"]
        counts = {}
        for split, rows in splits.items():
            counts[split] = len(rows)
            for row in rows:
                key = hashlib.sha256(bytes(row["map"])).hexdigest()
                if (
                    key in seen
                    or key != row["map_id"]
                    or sorted(row["order"]) != list(range(8))
                ):
                    raise ValueError(f"Overlapping or invalid map in {p}")
                seen.add(key)
                state = row["start"]
                for expected in row["states"]:
                    state = row["map"][state]
                    if state != expected:
                        raise ValueError("Incorrect map labels")
        datasets[str(p)] = dict(sha256=digest(p), splits=counts)
    for folder in ("data", "registry"):
        for p in sorted((root / folder).glob("*")):
            if p.is_file() and str(p) not in datasets:
                datasets[str(p)] = {"sha256": digest(p)}
    reports = {}
    for p in sorted((root / "experiments").glob("*.json")):
        name = p.stem
        result = root / "results" / name / "checksums.json"
        reports[name] = dict(
            registration=digest(p), archive=digest(result) if result.exists() else None
        )
    sessions = {}
    for inventory_path in sorted((root / "sessions").glob("*/*/checksums.json")):
        folder = inventory_path.parent
        registration = folder / "registration.json"
        if not registration.exists():
            continue
        study = json.loads(registration.read_bytes())["id"]
        if digest(registration) != reports[study]["registration"]:
            raise ValueError(f"Recovered study registration changed: {folder}")
        sessions[str(folder.relative_to(root / "sessions"))] = {
            "archive": digest(inventory_path),
            "registration": digest(registration),
        }
    return dict(datasets=datasets, experiments=reports, session_results=sessions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    path = Path("research/registry.json")
    obj = inventory()
    old = json.loads(path.read_text()) if path.exists() else {}
    if args.update:
        for kind, table in old.items():
            for key, values in table.items():
                for field, value in values.items():
                    if value is not None and obj[kind][key][field] != value:
                        raise ValueError(f"Existing identity changed: {key}")
        path.write_text(json.dumps(obj, indent=2) + "\n")
    elif obj != old:
        raise ValueError("Registry mismatch")
    print(
        len(obj["experiments"]),
        "registrations,",
        len(obj["datasets"]),
        "dataset identities",
    )
