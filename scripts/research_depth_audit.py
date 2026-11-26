"""Recompute paired depth-study measurements from the archived token streams."""

import re
from pathlib import Path

from loopqwen.evaluation.statistics import exact_mcnemar


def verify(root, decision, read, sha, require, number):
    reg = read(root / "registration.json")
    short = reg["id"] == "short-depth-transfer-01"
    panel = Path(
        reg["panel"]
        if short
        else reg.get("material", "research/data/deduction-stories-01.json")
    )
    require(
        sha(panel) == decision["data"]
        and sha(root / "registration.json") == decision["registration"],
        "Unbound depth evidence",
    )
    if decision["status"] == "interrupted":
        require(
            not decision["passed"] and bool(decision["reason"]),
            "Interrupted depth study promoted",
        )
        return
    data = read(panel)
    native = reg["id"].startswith(("native-induction-", "native-kinship-"))
    if native:
        forbidden = {
            row["story"]
            for filename in reg.get("exclude", [])
            for group in read(Path(filename)).values()
            for row in group
        }
        require(
            all(
                row["story"] not in forbidden
                for group in data.values()
                for row in group
            ),
            "Native panel overlaps a preceding study",
        )
        if reg.get("task") == "kinship":
            templates = [row["template"] for group in data.values() for row in group]
            require(
                len(templates) == len(set(templates)), "Family story template repeated"
            )
    if not short:
        stories = [r["story"] for group in data.values() for r in group]
        require(
            len(stories) == len(set(stories)),
            "Repeated story across losses or evaluation",
        )
        data = data[reg["panel"]]
    scores = {}
    arms = ("recurrent", "dense") if short else reg["arms"]
    outputs = {}
    for arm in arms:
        archive = read(root / (arm + ".json.gz"))
        rows = archive["records"]
        outputs[arm] = rows
        require(len(rows) == len(data), "Incomplete paired study")
        for r, item in zip(rows, data, strict=True):
            require(
                r["id"] == item["id"]
                and r["prompt_ids"] == item["prompt_ids"]
                and 0 < len(r["token_ids"]) <= reg["generation_cap"],
                "Changed prompts or budget",
            )
            tokens = r["token_ids"]
            complete = (
                tokens[-1] in (248044, 248046)
                and tokens.count(248069) == 1
                and 248068 not in tokens
            )
            require(complete == r["complete"], "Incorrect first-turn completion")
            if short:
                pred = number(tokens, r["text"])
                gold = item["gold"]
            else:
                if reg.get("task") == "kinship":
                    from loopqwen.research.kinship_data import WORDS as words
                    from loopqwen.research.kinship_data import answer
                elif reg["id"].startswith("native-induction-"):
                    from loopqwen.research.induction_data import WORDS as words
                    from loopqwen.research.induction_data import answer
                else:
                    from loopqwen.research.deduction_data import answer

                    words = ("cat", "mouse", "wolf", "sheep")

                body = re.split(r"<\|(?:im_end|endoftext)\|>$", r["text"])[0]
                args = (
                    (item["question"],)
                    if reg["id"].startswith(("native-induction-", "native-kinship-"))
                    else ()
                )
                pred = answer(body.split("</think>")[1], *args) if complete else None
                gold = item["answer"]
                require(
                    r["probe"]
                    == words[max(range(len(words)), key=r["probe_logits"].__getitem__)],
                    "Wrong probe",
                )
                require(
                    r["end_to_end"] >= r["seconds"] > 0,
                    "Invalid actual generation time",
                )
            require(
                pred == r["answer"] and r["correct"] == (pred == gold),
                "Wrong graded answer",
            )
        scores[arm] = sum(r["correct"] for r in rows)
        if short:
            require(
                archive["weights"] == sha(Path(reg[arm]) / "adapters.safetensors"),
                "Changed trained lineage",
            )
        elif reg["arms"][arm].get("fit"):
            folder = (
                root / arm
                if sum(bool(x.get("fit")) for x in reg["arms"].values()) > 1
                else root
            )
            model = read(folder / "reader.json")
            require(
                model["weights"] == sha(folder / "reader.safetensors")
                and model["updates"] == reg["updates"],
                "Wrong trained reader",
            )

    if native:
        counts = set()
        for arm, desc in reg["arms"].items():
            if not desc["depth"]:
                continue
            if desc.get("fit"):
                path = (
                    root / arm
                    if sum(bool(x.get("fit")) for x in reg["arms"].values()) > 1
                    else root
                )
            else:
                path = Path(desc["reader"])
                require(
                    sha(path / "reader.safetensors") == desc["weights"],
                    "Different fixed native layer weights",
                )
            meta = read(path / "reader.json")
            require(
                meta["fit_seed"] == reg["fit_seed"]
                and meta["updates"] == reg["updates"]
                and meta["rank"] == reg["rank"],
                "Unmatched native training configuration",
            )
            counts.add(meta["trainable_parameters"])
            if meta.get("architecture") == "anchored_native":
                for row, item in zip(outputs[arm], data, strict=True):
                    stages = row["support_probabilities"]
                    require(
                        len(stages) == 3
                        and all(
                            len(v) == len(item["facts"]) + 1
                            and abs(sum(v) - 1) < 1e-4
                            and all(0 <= x <= 1 for x in v)
                            for v in stages
                        ),
                        "Invalid native attention evidence",
                    )
        require(len(counts) == 1, "Native pair has unequal trainable capacity")

    def paired(group, first, second):
        diff = [
            int(a["correct"]) - int(b["correct"])
            for a, b, i in zip(outputs[first], outputs[second], data, strict=True)
            if group == "all" or i.get("source") == group
        ]
        return diff.count(1), diff.count(-1)

    a, b = arms
    wins, losses = paired("all", a, b)
    if short:
        for group, result in decision["scores"].items():
            require(
                (result["wins"], result["losses"]) == paired(group, a, b)
                and result["p"] == exact_mcnemar(result["wins"], result["losses"]),
                "Unpaired arithmetic statistics",
            )
            for arm in arms:
                subset = [
                    r
                    for r, i in zip(outputs[arm], data, strict=True)
                    if group == "all" or i["source"] == group
                ]
                require(
                    result[arm]["correct"] == sum(r["correct"] for r in subset)
                    and result[arm]["seconds"] == sum(r["seconds"] for r in subset),
                    "Arithmetic count or time",
                )
        v = decision["scores"]["all"]
        passed = (
            all(
                x["wins"] - x["losses"] >= 2
                for g, x in decision["scores"].items()
                if g != "all"
            )
            and v["p"] < 0.05
            and v[a]["seconds"] <= 1.3 * v[b]["seconds"]
            and decision["retention"][a] <= decision["retention"][b] + 0.03
        )
    else:
        for arm in arms:
            rows = outputs[arm]
            expected = {
                "correct": scores[arm],
                "complete": sum(r["complete"] for r in rows),
                "probe": sum(
                    r["probe"] == x["answer"] for r, x in zip(rows, data, strict=True)
                ),
                "seconds": sum(r["end_to_end"] for r in rows),
                "tokens": sum(len(r["token_ids"]) for r in rows),
            }
            require(decision["scores"][arm] == expected, "Invalid deduction summary")
        u, v = decision["scores"][a], decision["scores"][b]
        require(
            (wins, losses) == (decision["wins"], decision["losses"])
            and decision["p"] == exact_mcnemar(wins, losses),
            "Invalid paired deduction decision",
        )
        passed = (
            wins - losses >= reg["net_floor"]
            and decision["p"] < 0.05
            and u["probe"] >= v["probe"] + reg["probe_floor"]
            and u["seconds"] <= reg["time_ratio"] * v["seconds"]
        )
    if "pairs" in reg:
        counts = {}
        paired_results = {}
        for arm in arms:
            pairs = {}
            for r, item in zip(outputs[arm], data, strict=True):
                pairs.setdefault(item["pair"], []).append((r, item))
            require(len(pairs) == reg["pairs"], "Invalid intervention count")
            for group in pairs.values():
                (u, x), (v, y) = group
                positions = [
                    i
                    for i, (a0, b0) in enumerate(
                        zip(x["prompt_ids"], y["prompt_ids"], strict=True)
                    )
                    if a0 != b0
                ]
                require(
                    x["question"] == y["question"]
                    and x["answer"] != y["answer"]
                    and len(positions) == 2
                    and positions == x["changed_positions"] == y["changed_positions"]
                    and [x["prompt_ids"][i] for i in positions]
                    == [y["prompt_ids"][i] for i in reversed(positions)],
                    "Not a controlled token permutation",
                )
            paired_results[arm] = {
                key: all(r["correct"] for r, _ in pair) for key, pair in pairs.items()
            }
            counts[arm] = sum(paired_results[arm].values())
        x, y = counts.values()
        require(decision["both_correct"] == counts, "Wrong paired reassignments")
        passed &= x >= reg["pair_floor"] and x - y >= reg["pair_net"]
        if reg.get("class_swap"):
            from loopqwen.research.induction_counterfactual import bindings

            for row in data:
                _, examples = bindings(row["facts"], row["question"])
                require(
                    len(examples) == 1 and examples[0][2] == row["answer"],
                    "Ambiguous class intervention",
                )
            left, right = paired_results.values()
            wins = sum(value and not right[k] for k, value in left.items())
            losses = sum(value and not left[k] for k, value in right.items())
            require(
                decision["pair_p"] == exact_mcnemar(wins, losses),
                "Incorrect independent pair significance",
            )
            passed &= decision["pair_p"] < 0.05
    require(passed == decision["passed"], "Incorrect promotion decision")


def verify_release(root, decision, read, sha, require):
    from loopqwen.identity import fingerprint

    spec = read(root / "registration.json")
    require(
        sha(root / "registration.json") == decision["registration"],
        "Release registration mismatch",
    )
    for name, h in spec["prerequisites"].items():
        source = Path("research/results") / name
        require(
            sha(source / "checksums.json") == h
            and read(source / "decision.json")["passed"],
            "Invalid release prerequisite",
        )
    if decision["status"] != "completed":
        require(
            not decision["passed"] and decision["reason"], "Failed release promoted"
        )
        return
    folder = root / "model"
    policy = read(folder / "context-policy.json")
    require(
        sha(folder / "context-policy.json") == decision["policy"]
        and policy["schema"] == (4 if "induction" in spec else 3),
        "Wrong shipping policy",
    )
    for name, h in policy["files"].items():
        require(sha(folder / name) == h, "Changed release file: " + name)
    old = read(
        Path(spec.get("parent", spec["combined"])) / "model" / "context-policy.json"
    )
    require(
        all(policy["files"].get(k) == v for k, v in old["files"].items()),
        "Changed qualified component",
    )
    require(
        sha(folder / "evidence/reader.safetensors")
        == sha(Path(spec["reader"]) / "reader.safetensors"),
        "Different reasoning reader",
    )
    suites = [
        (
            "deduction",
            read(Path(spec["material"]))["replication"],
            Path(spec["longbase"]) / "recurrent.json.gz",
        ),
        (
            "mixed",
            read(Path("research/data/conditional-language-02.json")),
            Path(spec["combined"]) / "candidate.json.gz",
        ),
    ]
    if "induction" in spec:
        source = spec["induction"]
        suites.append(
            (
                "native_induction",
                read(Path(source["material"]))["replication"],
                Path(source["longbase"]) / "recurrent.json.gz",
            )
        )
        require(
            sha(folder / "induction/reader.safetensors")
            == sha(Path(source["reader"]) / "reader.safetensors"),
            "Wrong native recurrent weights",
        )
        audit = read(root / "export-verification.json")
        provenance = read(root / "export-provenance.json")
        require(
            audit["fresh_process_verified"]
            and audit["induction_probe"]["tokens"]
            and audit["evidence_probe"]["tokens"]
            and provenance["files"]["context-policy.json"] == decision["policy"]
            and read(root / "export-config.json")["context_policy_sha256"]
            == decision["policy"],
            "Invalid standalone export verification",
        )
    for label, rows, ref in suites:
        samples = read(root / (label + ".json.gz"))["records"]
        for r, v, item in zip(samples, read(ref)["records"], rows, strict=True):
            require(
                r["input_sha256"] == fingerprint(item["prompt_ids"])
                and r["id"] == item["id"]
                and r["token_ids"] == v["token_ids"]
                and r["route"] == (v["context_route"] if label == "mixed" else label),
                "Changed deployed answer",
            )
        require(
            decision[label]
            == {
                "identical": len(samples),
                "seconds": sum(r["elapsed_s"] for r in samples),
            },
            "Replay totals",
        )
    require(
        decision["raw_requests"] == 120
        and decision["retention"]
        == read(Path(spec["combined"]) / "decision.json")["retention"]["base"],
        "Raw language retention changed",
    )
    rows = read(Path("research/data/conditional-language-128k.json"))["panel"]
    ref = Path(spec["context"])
    records = read(root / "context.json.gz")["records"]
    for r, prior, row in zip(
        records, read(ref / "candidate.json.gz")["records"], rows, strict=True
    ):
        require(
            r["input_sha256"] == fingerprint(row["prompt_ids"])
            and r["token_ids"] == prior["token_ids"]
            and r["correct"]
            and r["gold"] == row["gold"]
            and r["context_route"] == prior["context_route"]
            and r["prompt_tokens"] == len(row["prompt_ids"]) >= 131072
            and r["attention_memory"]["start"]
            <= row["gold_start"]
            < row["gold_stop"]
            <= r["attention_memory"]["stop"],
            "Invalid long context replay",
        )
    seconds = sum(r["elapsed_s"] for r in records)
    require(
        decision["context"] == {"correct": 2, "seconds": seconds}
        and decision["passed"]
        == (seconds <= 1.75 * read(ref / "decision.json")["scores"]["base"]["seconds"]),
        "Wrong full-history decision",
    )


def verify_science(root, d, read, sha, require):
    from loopqwen.research.science_data import LETTERS, answer

    reg = read(root / "registration.json")
    panel = Path("research/data/science-composition-01.json")
    require(
        sha(panel) == d["data"]
        and sha(root / "registration.json") == d["registration"],
        "Wrong science cohort",
    )
    if d["status"] != "completed":
        require(not d["passed"] and d["reason"], "Incomplete study promoted")
        return
    rows = read(panel)[reg["panel"]]
    arms = {}
    for name, conf in reg["arms"].items():
        records = read(root / (name + ".json.gz"))["records"]
        if conf.get("fit"):
            require(
                sha(root / "reader.safetensors")
                == read(root / "reader.json")["weights"],
                "Science reader changed",
            )
        for item, r in zip(rows, records, strict=True):
            tokens = r["token_ids"]
            complete = (
                tokens[-1] in (248046, 248044)
                and tokens.count(248069) == 1
                and 248068 not in tokens
            )
            text = (
                r["text"]
                .split("</think>")[-1]
                .split("<|im_end|>")[0]
                .split("<|endoftext|>")[0]
            )
            pred = answer(text, item["options"]) if complete else None
            require(
                item["id"] == r["id"]
                and item["prompt_ids"] == r["prompt_ids"]
                and complete == r["complete"]
                and pred == r["answer"]
                and (pred == item["answer"]) == r["correct"]
                and r["probe"]
                == LETTERS[max(range(8), key=r["probe_logits"].__getitem__)],
                "Invalid science answer",
            )
        stats = {
            "correct": sum(v["correct"] for v in records),
            "complete": sum(v["complete"] for v in records),
            "probe": sum(
                v["probe"] == r["answer"] for v, r in zip(records, rows, strict=True)
            ),
            "seconds": sum(v["end_to_end"] for v in records),
        }
        require(d["scores"][name] == stats, "Wrong science totals")
        arms[name] = records
    a, b = arms
    delta = [
        int(x["correct"]) - int(y["correct"])
        for x, y in zip(arms[a], arms[b], strict=True)
    ]
    wins, losses = delta.count(1), delta.count(-1)
    p = exact_mcnemar(wins, losses)
    aa, bb = d["scores"][a], d["scores"][b]
    require(
        (d["wins"], d["losses"], d["p"]) == (wins, losses, p)
        and d["passed"]
        == (
            sum(delta) >= reg["net_floor"]
            and p < 0.05
            and aa["probe"] >= bb["probe"] + reg["probe_floor"]
            and aa["seconds"] <= bb["seconds"] * reg["time_ratio"]
        ),
        "Invalid science decision",
    )
