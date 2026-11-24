"""Seal an immutable measurement directory and independently recount its task outputs."""

import argparse
import gzip
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loopqwen.evaluation.answers import numeric


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(
        gzip.decompress(path.read_bytes())
        if path.suffix == ".gz"
        else path.read_bytes()
    )


def require(value, detail):
    if not value:
        raise ValueError(detail)


def trajectory(row, depth):
    node = row["start"]
    gold = []
    for _ in range(depth):
        node = row["map"][node]
        gold.append(node)
    return gold


def trace_score(row):
    match = re.fullmatch(
        r"\s*([0-7])\s*(?:,|->|→)\s*([0-7])\s*(?:,|->|→)\s*([0-7])[.!]?\s*",
        row["visible"],
    )
    return bool(
        match
        and row["stop"] != "cap"
        and list(map(int, match.groups())) == trajectory(row, 3)
    )


def checksums(root):
    return {
        str(p.relative_to(root)): sha(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p != root / "checksums.json"
    }


def verify(root):
    index = read(root / "checksums.json")
    require(index == checksums(root), "Archive integrity")
    decision = read(root / "decision.json")
    study = read(root / "registration.json")["id"]
    if (root / "parent.json.gz").exists() or study == "trace-depth-01":
        scores = decision.get("scores", decision)
        arms = scores.keys() if "scores" in decision else ("parent", "child")
        keys = None
        for arm in arms:
            rows = read(root / (arm + ".json.gz"))["records"]
            inputs = [(r["id"], r["prompt_ids"]) for r in rows]
            require(keys is None or keys == inputs, "Different trace requests")
            keys = inputs
            verified = [trace_score(r) for r in rows]
            require(
                verified == [r["score"]["verified"] for r in rows],
                "Invalid trace certificate",
            )
            require(sum(verified) == scores[arm]["verified"], "Incorrect trace summary")
    if study == "external-anchor-01":
        cohort = read(Path("research/data/external-anchor.json"))
        gold = {r["id"]: r["gold"] for group in ("gsm", "arc") for r in cohort[group]}
        inputs = []
        for arm in ("base", "candidate"):
            rows = read(root / (arm + ".json.gz"))["records"]
            inputs.append(
                [(r["id"], r.get("prompt_ids", r.get("streams"))) for r in rows]
            )
            for group in ("gsm", "arc"):
                count = 0
                for r in (v for v in rows if v["family"] == group):
                    target = gold[r["id"]]
                    ok = (
                        (r["complete"] and numeric(r["visible"]) == numeric(target))
                        if group == "gsm"
                        else max(
                            range(len(r["logp"])),
                            key=lambda i: r["logp"][i]
                            / (len(r["streams"][i]["ids"]) - r["streams"][i]["prefix"]),
                        )
                        == target
                    )
                    require(ok == r["correct"], "External verdict")
                    count += ok
                require(count == decision[arm][group]["correct"], "External aggregate")
        require(inputs[0] == inputs[1], "Unpaired external evaluation")
    if study == "turn-calibration-01":
        rows = read(root / "generations.json.gz")["records"]
        answers = [numeric(r["visible"]) if r["complete"] else None for r in rows]
        require(
            sum(a is not None for a in answers) == decision["valid"],
            "Calibration completion",
        )
        require(
            sum(a == numeric(r["item"]["gold"]) for a, r in zip(answers, rows))
            == decision["correct"],
            "Calibration accuracy",
        )
    if study == "functional-zero-01":
        for kind in ("functional", "permutation"):
            rows = read(root / (kind + ".json.gz"))["records"]
            exact = [0] * 6
            paths = [0] * 6
            for r in rows:
                gold = trajectory(r, 6)
                predicted = [
                    max(range(8), key=p.__getitem__) for p in r["probabilities"]
                ]
                require(
                    gold == r["states"] and predicted == r["predicted"],
                    "Functional state mismatch",
                )
                for j in range(6):
                    exact[j] += gold[j] == predicted[j]
                    paths[j] += gold[: j + 1] == predicted[: j + 1]
            require(
                exact == decision[kind]["correct"]
                and paths == decision[kind]["trajectory"],
                "Functional summary",
            )
    print(study, len(index), "files verified")


def publish(name):
    root = Path("research/results") / name
    src = Path("outputs/research") / name
    if root.exists():
        return verify(root)
    require((src / "decision.json").is_file(), "Missing stage decision")
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent) as temp:
        dest = Path(temp) / "archive"
        shutil.copytree(src, dest)
        shutil.copy(
            Path("research/experiments") / (name + ".json"), dest / "registration.json"
        )
        (dest / "checksums.json").write_text(
            json.dumps(checksums(dest), indent=2) + "\n"
        )
        verify(dest)
        dest.rename(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(Path("research/results") / args.name)
    else:
        publish(args.name)
