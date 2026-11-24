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
    if study == "functional-interleave-01":
        verify_interleave(root, decision)
    if study == "gsm-equations-01":
        verify_math(root, decision)
    if study == "functional-replay-cap84":
        verify_replay(root, decision)
    if study == "functional-terminal-01":
        verify_terminal(root, decision)
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
    if study == "verified-generations-01":
        verify_generations(root, decision)
    print(study, len(index), "files verified")


def verify_interleave(root, decision):
    reg = read(root / "registration.json")
    material = read(Path("research/data") / (reg["id"] + ".json"))
    dataset = read(Path("research/data") / (reg["dataset"] + ".json.gz"))["splits"]
    require(material["fresh"] == dataset["pool1"], "Fresh curriculum differs")
    reserved = set(material["dev_ids"])
    require(
        not any(
            r["map_id"] in reserved
            for name in ("fresh", "control", "permutation")
            for r in material[name]
        ),
        "Replay leakage",
    )
    for arm in ("control", "permutation"):
        folder = root / arm
        meta = read(folder / "model/state.json")
        require(
            meta["parent"] == reg["parent_sha256"] and meta["step"] == 16,
            "Interleaved checkpoint lineage",
        )
        for j, h in enumerate(meta["history"]):
            want = (
                material["fresh"][6 * j : 6 * j + 6] + material[arm][2 * j : 2 * j + 2]
            )
            require(
                [f["id"] for f in h["feedback"]] == [r["map_id"] for r in want],
                "Replay membership",
            )
        rows = read(folder / "eval.json.gz")["records"]
        hits = [0] * 6
        paths = [0] * 6
        for r, gold in zip(rows, dataset["dev"], strict=True):
            require(r["map_id"] == gold["map_id"], "Wrong replay evaluation map")
            expected = trajectory(gold, 6)
            pred = [p.index(max(p)) for p in r["probabilities"]]
            for j in range(6):
                hits[j] += pred[j] == expected[j]
                paths[j] += pred[: j + 1] == expected[: j + 1]
        require(
            hits == decision[arm]["functional"]["correct"]
            and paths == decision[arm]["functional"]["trajectory"],
            "Wrong interleave accuracy",
        )
    parent = decision["parent"]
    mixed = decision["permutation"]
    require(
        decision["passed"]
        == (
            (mixed["functional"]["correct"][-1] - parent["functional"]["correct"][-1])
            / len(dataset["dev"])
            >= 0.10
            and parent["anchor"][-1] - mixed["anchor"][-1] <= 2
        ),
        "Interleave gate",
    )


def verify_math(root, decision):
    data = read(Path("research/data/gsm-equations-01.json"))
    streams = []
    answers = []
    for arm in ("base", "candidate"):
        report = read(root / (arm + ".json.gz"))
        records = report["records"]
        require(
            report["cohort"] == sha(Path("research/data/gsm-equations-01.json")),
            "Math cohort digest",
        )
        flags = []
        valid = 0
        for r, item in zip(records, data["dev"], strict=True):
            require(
                (r["id"], r["gold"]) == (item["id"], item["gold"]), "Wrong math item"
            )
            final = numeric(r["visible"]) if r["complete"] else None
            require(final == r["answer"], "Wrong visible answer")
            ok = final == item["gold"]
            require(ok == r["correct"], "Wrong completed answer")
            valid += final is not None
            flags.append(ok)
        require(
            (sum(flags), valid) == (decision[arm]["correct"], decision[arm]["valid"]),
            "Math totals",
        )
        streams.append([r["prompt_ids"] for r in records])
        answers.append(flags)
    require(streams[0] == streams[1], "Unpaired math questions")
    diff = [int(b) - int(a) for a, b in zip(*answers, strict=True)]
    require(
        (diff.count(1), diff.count(-1))
        == (decision["paired"]["wins"], decision["paired"]["losses"]),
        "Math paired totals",
    )
    passed = (
        decision["completed_updates"] == read(root / "registration.json")["updates"]
        and sum(diff) >= 2
        and decision["candidate"]["valid"] >= 16
        and decision["anchor_before"][-1] - decision["anchor_after"][-1] <= 2
    )
    require(passed == decision["passed"], "Math gate")


def verify_replay(root, decision):
    reg = read(root / "registration.json")
    groups = read(Path("research/data") / (reg["dataset"] + ".json.gz"))["splits"]

    def counts(path):
        rows = read(path)["records"]
        hits = [0] * 6
        for r, expected in zip(rows, groups["dev"], strict=True):
            require(r["map_id"] == expected["map_id"], "Wrong functional cohort")
            pred = [p.index(max(p)) for p in r["probabilities"]]
            require(pred == r["predicted"], "Invalid categorical argmax")
            for i, (p, g) in enumerate(zip(pred, trajectory(expected, 6), strict=True)):
                hits[i] += p == g
        return hits

    require(
        counts(root / "initial.json.gz") == decision["parent"]["functional"]["correct"],
        "Initial accuracy",
    )
    prev = {k: reg["parent_sha256"] for k in ("static", "online")}
    for j, arms in enumerate(decision["rounds"], 1):
        pool = groups[f"pool{j}"]
        for arm in prev:
            folder = root / f"round{j}" / arm
            require(
                counts(folder / "eval.json.gz") == arms[arm]["functional"]["correct"],
                "Round accuracy",
            )
            meta = read(folder / "model/state.json")
            require(
                meta["parent"] == prev[arm]
                and meta["step"] == j * len(meta["history"]),
                "Replay lineage",
            )
            prev[arm] = sha(folder / "model/adapters.safetensors")
            require(prev[arm] == arms[arm]["weights"], "Replay weights")
            feedback = [r for h in meta["history"] for r in h["feedback"]]
            require(
                [r["id"] for r in feedback[: len(pool)]] == [r["map_id"] for r in pool],
                "Fresh schedule",
            )
            selected = arms[arm]["selected"]
            require(
                [r["id"] for r in feedback[len(pool) :]] == selected
                and len(selected) == reg["replay_per_round"],
                "Replay schedule",
            )
            for i, r in enumerate(feedback[: len(pool)]):
                gold = trajectory(pool[i], 6)
                prefix = next(
                    (
                        k
                        for k, (a, b) in enumerate(
                            zip(gold, r["predicted"], strict=True)
                        )
                        if a != b
                    ),
                    6,
                )
                require(prefix == r["prefix"], "Wrong online prefix length")
            for slot, item in enumerate(selected):

                def priority(row):
                    prefix = row["prefix"]
                    rank = (
                        0
                        if arm == "static"
                        else (
                            prefix
                            if slot % 5 == 3
                            else -prefix if slot % 5 == 4 else abs(prefix - 3)
                        )
                    )
                    tie = hashlib.sha256(
                        json.dumps(
                            [reg["seed"] + j, row["id"]], separators=(",", ":")
                        ).encode()
                    ).hexdigest()
                    return rank, tie

                require(
                    item == min(feedback[3 * slot : 3 * slot + 3], key=priority)["id"],
                    "Unreconstructible curriculum",
                )


def verify_terminal(root, decision):
    reg = read(root / "registration.json")
    cohort = read(Path("research/data") / (reg["id"] + ".json.gz"))["splits"]["dev"]
    inputs = []
    outcomes = []
    for depth in reg["depths"]:
        arm = f"K{depth}"
        output = read(root / arm / "eval.json.gz")
        require(
            output["weights"] == sha(root / arm / "model/adapters.safetensors"),
            "Wrong trained adapter",
        )
        records = output["records"]
        require(
            len(records) == len(cohort) == decision[arm]["n"], "Wrong terminal cohort"
        )
        correct = []
        for r, expected in zip(records, cohort, strict=True):
            require(
                all(r[k] == expected[k] for k in ("map_id", "map", "start", "order")),
                "Wrong terminal request",
            )
            require(
                r["stop"] in ("line", "eos", "chat_end", "cap")
                and r["complete"] == (r["stop"] != "cap"),
                "Wrong completion verdict",
            )
            ok = r["complete"] and r["visible"].strip() == str(
                trajectory(expected, 3)[-1]
            )
            require(ok == r["correct"], "Wrong terminal verdict")
            correct.append(ok)
        inputs.append([r["prompt_ids"] for r in records])
        outcomes.append(correct)
        require(sum(correct) == decision[arm]["correct"], "Wrong terminal total")
        require(
            sum(r["complete"] for r in records) == decision[arm]["completed"],
            "Wrong completion count",
        )
        require(
            sum(len(r["token_ids"]) for r in records) == decision[arm]["tokens"],
            "Wrong token count",
        )
    require(inputs[0] == inputs[1], "Unpaired terminal prompts")
    diff = [int(b) - int(a) for a, b in zip(*outcomes, strict=True)]
    require(
        diff.count(1) == decision["paired"]["wins"]
        and diff.count(-1) == decision["paired"]["losses"],
        "Wrong paired counts",
    )
    retained = (
        decision["parent_categorical"][-1] - decision["K3"]["categorical"][-1] <= 2
    )
    require(
        decision["passed"] == (sum(diff) / len(cohort) >= 0.05 and retained),
        "Wrong terminal gate",
    )


def verify_generations(root, decision):
    reg = read(root / "registration.json")
    groups = read(Path("research/data") / (reg["id"] + ".json.gz"))["splits"]
    previous = {arm: reg["parent_sha256"] for arm in ("self", "gold")}
    dev = [r["map_id"] for r in groups["dev"]]

    def score(path, summary):
        records = read(path)["records"]
        require([r["map_id"] for r in records] == dev, "Wrong evaluation cohort")
        require(
            sum(map(trace_score, records)) == summary["verified"],
            "Incorrect verified generation count",
        )

    score(root / "parent-eval.json.gz", decision["parent_eval"])
    for round_id, stage in decision["stages"].items():
        folder = root / ("round" + round_id)
        selection = read(folder / "selection.json")
        pool = groups["pool" + round_id]
        require(
            sha(folder / "selection.json") == stage["selection"], "Selection digest"
        )
        require(selection["parent"] == previous["self"], "Wrong teacher")
        rolls = read(folder / "candidates.json.gz")["records"]
        require(
            [r["map_id"] for r in rolls]
            == [r["map_id"] for r in pool for _ in range(reg["samples_per_map"])],
            "Rollout cohort",
        )
        first = {}
        accepted = 0
        for r in rolls:
            valid = trace_score(r)
            require(valid == r["score"]["verified"], "Rollout verdict disagrees")
            if valid:
                accepted += 1
                first.setdefault(r["map_id"], r["token_ids"])
        require(accepted == stage["yield_summary"]["accepted"], "Verified yield")
        require(
            list(first)[: reg["select"]] == selection["self_ids"],
            "Wrong self curriculum",
        )
        require(
            selection["responses"] == {i: first[i] for i in selection["self_ids"]},
            "Changed sampled answer",
        )
        require(
            [r["map_id"] for r in pool[: reg["select"]]] == selection["gold_ids"],
            "Changed static curriculum",
        )
        for arm in ("self", "gold"):
            if arm in stage:
                require(stage[arm]["parent"] == previous[arm], "Wrong training lineage")
                previous[arm] = sha(folder / arm / "adapters.safetensors")
                require(
                    previous[arm] == stage[arm]["weights"], "Wrong generation weights"
                )
                score(folder / (arm + "-eval.json.gz"), stage[arm]["score"])


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
