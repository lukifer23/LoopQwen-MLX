"""Archive and independently verify the bounded curriculum experiment using Python alone."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory


def read(p):
    return json.loads(
        gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
    )


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(root):
    inventory = read(root / "checksums.json")
    if inventory != {
        str(p.relative_to(root)): sha(p)
        for p in root.rglob("*")
        if p.is_file() and p != root / "checksums.json"
    }:
        raise ValueError("Archive hash mismatch")
    spec = read(root / "registration.json")
    splits = read(root / "pools.json.gz")
    ids = [r["map_id"] for rows in splits.values() for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Map split overlap")
    board = read(root / "scoreboard.json")
    for file in sorted(root.glob("*.json.gz")):
        if file.name == "pools.json.gz":
            continue
        result = read(file)
        rows = result["records"]
        split = (
            "pool" + file.name.split("-")[0][-1]
            if "diagnosis" in file.name
            else "development"
        )
        expected = {(r["map_id"], o) for r in splits[split] for o in spec["orders"]}
        observed = set()
        correct = Counter()
        paths = Counter()
        for r in rows:
            h = hashlib.sha256(bytes(r["map"])).hexdigest()
            if h != r["map_id"] or sorted(r["order"]) != list(range(8)):
                raise ValueError("Map identity")
            observed.add((h, r["condition"]))
            node = r["start"]
            path = True
            for i, pr in enumerate(r["probabilities"]):
                node = r["map"][node]
                pred = max(range(8), key=pr.__getitem__)
                if (
                    node != r["gold"][i]
                    or pred != r["pred"][i]
                    or abs(sum(pr) - 1) > 1e-5
                ):
                    raise ValueError("State mismatch")
                path = path and pred == node
                correct[i] += pred == node
                paths[i] += path
        if expected != observed or len(expected) != len(rows):
            raise ValueError("Missing or repeated presentation")
        report = result["summary"]
        if [correct[i] / len(rows) for i in range(6)] != report["accuracy"] or [
            paths[i] / len(rows) for i in range(6)
        ] != report["path"]:
            raise ValueError("Score mismatch")
        if "evaluation" in file.name and report != board[file.name.split("-")[0]]:
            raise ValueError("Scoreboard differs")
    for p in root.glob("*-lineage.json"):
        name = p.stem.split("-")[0]
        j = int(name[-1])
        data = read(p)
        expect = (
            spec["weights"]
            if j == 1
            else read(root / (name[:-1] + "1-lineage.json"))["weights"]
        )
        if data["parent"] != expect or data["weights"] != sha(
            root / "models" / name / "adapters.safetensors"
        ):
            raise ValueError("Invalid lineage")
        selected = read(root / (name + "-curriculum.json"))
        rows = selected["selected"]
        if (
            selected["parent"] != expect
            or len({v["id"] for v in rows}) != 128
            or set(Counter(v["stratum"] for v in rows).values()) != {8}
            or not {v["id"] for v in rows}
            <= {m["map_id"] for m in splits["pool" + str(j)]}
        ):
            raise ValueError("Invalid training curriculum")
        if (
            name.startswith("adaptive")
            and read(root / (name + "-diagnosis.json.gz"))["weights"] != expect
        ):
            raise ValueError("Stale learner diagnosis")
    print("Verified bounded RSI archive:", sorted(board))


def publish(root):
    if root.exists():
        raise FileExistsError(root)
    src = Path("outputs/rsi1")
    with TemporaryDirectory(dir=root.parent) as tmp:
        build = Path(tmp) / "release"
        shutil.copytree(src, build, ignore=shutil.ignore_patterns("study.lock"))
        shutil.copy("experiments/rsi_pilot.json", build / "registration.json")
        for p in build.glob("*-lineage.json"):
            row = read(p)
            dst = build / "models" / p.stem.split("-")[0]
            dst.mkdir(parents=True)
            cp = Path(row["generation"])
            if sha(cp / "optimizer.safetensors") != row["optimizer"]:
                raise ValueError("Optimizer integrity")
            shutil.copy(cp / "adapters.safetensors", dst)
            shutil.copy(cp / "state.json", dst)
        (build / "checksums.json").write_text(
            json.dumps(
                {
                    str(p.relative_to(build)): sha(p)
                    for p in sorted(build.rglob("*"))
                    if p.is_file()
                },
                indent=2,
            )
            + "\n"
        )
        verify(build)
        build.rename(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--archive", type=Path, default=Path("docs/measurements/rsi1"))
    a = parser.parse_args()
    (publish if a.publish else verify)(a.archive)
