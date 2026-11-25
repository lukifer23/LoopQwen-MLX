"""Seal an immutable measurement directory and independently recount its task outputs."""

import argparse
import gzip
import hashlib
import json
import math
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
                            key=lambda i: (
                                r["logp"][i]
                                / (
                                    len(r["streams"][i]["ids"])
                                    - r["streams"][i]["prefix"]
                                )
                            ),
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
    if study in ("context-128k-01", "context-128k-disk"):
        verify_context(root, decision)
    if study == "rationale-transfer-01":
        verify_transfer(root, decision)
    if study in ("gsm-rationale-01", "gsm-rationale-endpoint"):
        verify_rationale(root, decision)
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


def verify_context(root, decision):
    from loopqwen.identity import fingerprint

    reg = read(root / "registration.json")
    prompt = read(root / "prompt.json.gz")
    ids = prompt["prompt_ids"]
    require(
        sha(root / "registration.json") == decision["registration"],
        "Context registration",
    )
    require(
        len(ids) == reg["prompt_tokens"] and prompt["input_sha256"] == fingerprint(ids),
        "Context prompt stream",
    )
    if "reference" in reg:
        old = read(Path("research/results") / reg["reference"] / "prompt.json.gz")
        require(old == prompt, "Disk variant changed the frozen prompt")
        parity = read(root / "parity.json") if (root / "parity.json").exists() else {}
        if decision["status"] == "parity_failed":
            require(
                any(x != 0 for x in parity[decision["arm"]])
                and not decision["capacity"],
                "Parity stop",
            )
            return
        if decision["completed"] or decision["status"] == "completed":
            require(
                parity == {a: [0.0] * 4 for a in ("base", "candidate")},
                "Missing logit parity",
            )
    completed = []
    for arm, weight in (
        ("base", decision["base_revision"]),
        ("candidate", reg["weights"]),
    ):
        path = root / f"{arm}.json"
        if not path.exists():
            continue
        row = read(path)
        require(
            row["weights"] == weight and row["input_sha256"] == fingerprint(ids),
            "Context arm identity",
        )
        require(
            row["prompt_tokens"] == len(ids)
            and 0 < len(row["token_ids"]) <= reg["generation_cap"]
            and row["elapsed_s"] >= row["ttft_s"] > 0,
            "Incomplete context record",
        )
        require(
            row["correct"] == (row["answer"] == prompt["gold"]),
            "Context retrieval score",
        )
        completed.append(arm)
    require(completed == decision["completed"], "Context arm inventory")
    require(
        decision["capacity"] == (completed == ["base", "candidate"]),
        "Context capacity gate",
    )
    if decision["capacity"]:
        require(
            decision["retrieval"]
            == {arm: read(root / f"{arm}.json")["correct"] for arm in completed},
            "Context retrieval aggregate",
        )
    else:
        require(
            decision["status"] == "interrupted"
            and decision["reason"]
            and 0 <= decision["processed_positions"] < len(ids) + reg["generation_cap"],
            "Context interruption",
        )


def verify_transfer(root, decision):
    from loopqwen import REVISION

    reg = read(root / "registration.json")
    cohort = Path("research/data") / (reg["id"] + ".json")
    require(
        sha(cohort) == decision["data"]
        and sha(root / "registration.json") == decision["registration"],
        "Transfer material",
    )
    items = read(cohort)
    require(
        len({r["id"] for r in items}) == len(items) == reg["per_source"] * 2,
        "Transfer cohort size",
    )
    reports = {}
    for arm, weight in (("base", REVISION), ("candidate", reg["weights"])):
        file = root / (arm + ".json.gz")
        if not file.exists():
            require(decision["status"] == "interrupted", "Missing transfer generations")
            continue
        report = read(file)
        require(report["weights"] == weight, "Transfer weights")
        rows = report["records"]
        require(len(rows) <= len(items), "Extra transfer generations")
        for row, item in zip(rows, items):
            require(
                all(row[k] == item[k] for k in ("id", "prompt_ids", "gold")),
                "Wrong transfer prompt",
            )
            tokens = row["token_ids"]
            close = [i for i, t in enumerate(tokens) if t == 248069]
            boundary = tokens[-1] in (248046, 248044)
            complete = boundary and len(close) == 1 and 248068 not in tokens
            require(
                row["complete"] == complete
                and 0 < len(tokens) <= reg["generation_cap"],
                "Invalid first turn",
            )
            answer = numeric(row["visible"]) if complete else None
            require(
                row["answer"] == answer and row["correct"] == (answer == item["gold"]),
                "Transfer numeric score",
            )
            require(
                0 < row["seconds_at_256"] <= row["seconds"], "Invalid prefix timing"
            )
        reports[arm] = rows
    if decision["status"] == "interrupted":
        require(
            not decision["passed"] and decision["reason"],
            "Interrupted transfer decision",
        )
        return
    require(
        all(len(rows) == len(items) for rows in reports.values()),
        "Missing transfer items",
    )
    for key, cap in (("full", reg["generation_cap"]), ("prefix_256", 256)):
        for source in ("all", *reg["sources"]):
            summary = decision[key][source]
            ix = [
                i
                for i, item in enumerate(items)
                if source == "all" or item["source"] == source
            ]
            flags = {}
            for arm, rows in reports.items():
                selected = [rows[i] for i in ix]
                flags[arm] = [
                    r["correct"] and len(r["token_ids"]) <= cap for r in selected
                ]
                stats = summary[arm]
                require(
                    stats["correct"] == sum(flags[arm])
                    and stats["valid"]
                    == sum(
                        r["answer"] is not None and len(r["token_ids"]) <= cap
                        for r in selected
                    ),
                    "Transfer score totals",
                )
                require(
                    stats["tokens"]
                    == sum(min(cap, len(r["token_ids"])) for r in selected)
                    and stats["seconds"]
                    == sum(
                        r["seconds_at_256"] if cap == 256 else r["seconds"]
                        for r in selected
                    ),
                    "Transfer cost totals",
                )
            delta = [
                int(c) - int(b)
                for c, b in zip(flags["candidate"], flags["base"], strict=True)
            ]
            wins, losses = delta.count(1), delta.count(-1)
            n = wins + losses
            p_value = min(
                1, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n
            )
            require(
                (summary["n"], summary["wins"], summary["losses"], summary["p"])
                == (len(ix), wins, losses, p_value),
                "Transfer paired totals",
            )
    full = decision["full"]
    passed = (
        all(full[src]["wins"] - full[src]["losses"] >= 2 for src in reg["sources"])
        and full["all"]["p"] < 0.05
        and full["all"]["candidate"]["seconds"] <= full["all"]["base"]["seconds"]
    )
    require(decision["passed"] == passed, "Transfer gate")


def verify_rationale(root, decision):
    from loopqwen import REVISION
    from loopqwen.identity import fingerprint

    reg = read(root / "registration.json")
    endpoint = reg.get("training_study")
    training = (
        read(Path("research/experiments") / (endpoint + ".json")) if endpoint else reg
    )
    manifest = Path("research/data") / (training["id"] + ".json")
    frozen = read(manifest)
    data = read(Path(training["data"]))["dev"]
    require(
        sha(root / "registration.json") == decision["registration"]
        and sha(manifest) == decision["data"],
        "Rationale registration",
    )
    require(
        [r["id"] for r in data] == frozen["dev_ids"]
        and len(set(frozen["train_ids"]) & set(frozen["dev_ids"])) == 0,
        "Rationale cohort",
    )
    model = root / "model"
    if model.exists():
        state = read(model / "state.json")
        require(
            sha(model / "adapters.safetensors") == decision["weights"]
            and state["parent"] == training["parent_sha256"]
            and state["step"] == reg["updates"],
            "Rationale checkpoint",
        )
    if endpoint:
        require(
            not decision["passed"] and decision["weights"] == reg["weights"],
            "Interrupted endpoint cannot promote",
        )
    answers = []
    for arm in ("base", "candidate"):
        path = root / (arm + ".json.gz")
        if not path.exists():
            require(arm not in decision, "Missing arm outputs")
            continue
        archive = read(path)
        require(
            archive["weights"] == (REVISION if arm == "base" else decision["weights"]),
            "Wrong GSM8K weights",
        )
        records = archive["records"]
        require(len(records) <= len(data), "Unregistered GSM8K example")
        flags = []
        for row, target in zip(records, data):
            require(
                (row["id"], row["gold"]) == (target["id"], target["gold"]),
                "Wrong GSM8K row",
            )
            visible = numeric(row["visible"]) if row["complete"] else None
            require(
                row["answer"] == visible
                and row["correct"] == (visible == target["gold"]),
                "GSM8K answer",
            )
            flags.append(row["correct"])
        if arm in decision:
            require(
                fingerprint([r["prompt_ids"] for r in records])
                == frozen["dev_prompts"],
                "GSM8K prompt stream",
            )
            stats = decision[arm]
            require(
                stats["correct"] == sum(flags)
                and stats["valid"] == sum(r["answer"] is not None for r in records)
                and stats["seconds"] == sum(r["seconds"] for r in records)
                and stats["tokens"] == sum(len(r["token_ids"]) for r in records),
                "GSM8K accuracy or cost",
            )
        answers.append(flags)
    if decision["status"] == "interrupted":
        require(
            not decision["passed"] and decision["reason"], "Incomplete rationale gate"
        )
        return
    diff = [int(c) - int(b) for b, c in zip(*answers, strict=True)]
    halves = [sum(diff[:16]), sum(diff[16:])]
    require(
        len(diff) == 32
        and (diff.count(1), diff.count(-1), halves)
        == (decision["wins"], decision["losses"], decision["half_deltas"]),
        "Paired GSM8K totals",
    )
    c, b = decision["candidate"], decision["base"]
    require(
        decision.get("descriptive_thresholds_met", decision["passed"])
        == (
            c["correct"] >= 8
            and min(halves) > 0
            and sum(diff) >= 4
            and c["seconds"] <= 1.25 * b["seconds"]
            and c["retention"]["nll"] <= b["retention"]["nll"] + 0.03
        ),
        "Rationale gate",
    )


def replay_outcomes(report, cohort, depth, weight, parent_records):
    """Rescore each exit for the same map, start and token stream across checkpoints."""
    require(report["weights"] == weight, "Wrong replay prediction weights")
    hits, paths, terminal = [0] * depth, [0] * depth, []
    for row, gold, parent in zip(
        report["records"], cohort, parent_records, strict=True
    ):
        require(
            all(
                row[k] == gold[k]
                for k in ("map", "map_id", "start", "example_id", "order")
            )
            and row["ids"] == parent["ids"],
            "Unpaired replay request",
        )
        expected = trajectory(gold, depth)
        predicted = [p.index(max(p)) for p in row["probabilities"]]
        require(
            predicted == row["predicted"] and row["states"][:depth] == expected,
            "Replayed state or target differs",
        )
        prefix_correct = True
        for j, (state, target) in enumerate(zip(predicted, expected, strict=True)):
            correct = state == target
            hits[j] += correct
            prefix_correct = prefix_correct and correct
            paths[j] += prefix_correct
        terminal.append(predicted[-1] == expected[-1])
    return hits, paths, terminal


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
    previous = Path("research/results/functional-replay-cap84")
    parent = read(previous / "initial.json.gz")
    parent_summary = read(previous / "decision.json")["parent"]
    require(decision["parent"] == parent_summary, "Wrong reference checkpoint scores")
    cohort = dataset["dev"]
    require(material["dev_ids"] == [r["map_id"] for r in cohort], "Wrong cohort order")
    _, _, parent_terminal = replay_outcomes(
        parent, cohort, reg["depth"], reg["parent_sha256"], parent["records"]
    )
    terminal = {"parent": parent_terminal}
    for arm in ("control", "permutation"):
        folder = root / arm
        meta = read(folder / "model/state.json")
        require(
            meta["parent"] == reg["parent_sha256"]
            and meta["step"] == len(meta["history"]) == reg["train_updates"],
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
        weight = sha(folder / "model/adapters.safetensors")
        require(weight == decision[arm]["weights"], "Wrong interleave checkpoint")
        hits, paths, terminal[arm] = replay_outcomes(
            read(folder / "eval.json.gz"),
            cohort,
            reg["depth"],
            weight,
            parent["records"],
        )
        require(
            hits == decision[arm]["functional"]["correct"]
            and paths == decision[arm]["functional"]["trajectory"],
            "Wrong interleave accuracy",
        )
    for arm in ("parent", "control"):
        pairs = list(zip(terminal["permutation"], terminal[arm], strict=True))
        paired = decision["paired"][arm]
        require(
            sum(a and not b for a, b in pairs) == paired["wins"]
            and sum(b and not a for a, b in pairs) == paired["losses"],
            "Incorrect paired interleave outcomes",
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

    initial = read(root / "initial.json.gz")

    def score(report, weight, summary):
        hits, paths, _ = replay_outcomes(
            report, groups["dev"], reg["depth"], weight, initial["records"]
        )
        require(
            hits == summary["correct"] and paths == summary["trajectory"],
            "Replay accuracy",
        )

    score(initial, reg["parent_sha256"], decision["parent"]["functional"])
    prev = {k: reg["parent_sha256"] for k in ("static", "online")}
    for j, arms in enumerate(decision["rounds"], 1):
        pool = groups[f"pool{j}"]
        for arm in prev:
            folder = root / f"round{j}" / arm
            score(
                read(folder / "eval.json.gz"),
                arms[arm]["weights"],
                arms[arm]["functional"],
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
                            else -prefix
                            if slot % 5 == 4
                            else abs(prefix - 3)
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


def publish(name, session=None):
    root = (
        Path("research/sessions") / session if session else Path("research/results")
    ) / name
    src = (
        Path("outputs/research/recovered") if session else Path("outputs/research")
    ) / name
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
    parser.add_argument(
        "--recovered-session",
        help="Archive a recovered primary run under this session date",
    )
    args = parser.parse_args()
    if args.recovered_session and not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}", args.recovered_session
    ):
        parser.error("Use an ISO session date")
    if args.verify:
        root = (
            Path("research/sessions") / args.recovered_session
            if args.recovered_session
            else Path("research/results")
        )
        verify(root / args.name)
    else:
        publish(args.name, args.recovered_session)
