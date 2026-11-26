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
    if study.startswith("folded-system-"):
        from loopqwen.research.folded_system import verify as verify_folded

        verify_folded(root, decision)
    if study.startswith("draft-program-"):
        from loopqwen.research.draft_program import verify as verify_drafts

        verify_drafts(root, decision)
    if study.startswith("meaning-alignment-"):
        from loopqwen.research.meaning_alignment import verify as verify_alignment

        verify_alignment(root, decision)
        return
    if study.startswith("session-tokenizer-"):
        from loopqwen.research.session_tokenizer import verify as verify_tokenizer

        verify_tokenizer(root, decision)
        return
    if study.startswith("packed-decode-full-"):
        from loopqwen.research.full_decode import verify as verify_full_decode

        verify_full_decode(root, decision)
        return
    if study.startswith("packed-decode-kernel-"):
        from loopqwen.research.packed_decode import verify as verify_packed_decode

        verify_packed_decode(root, decision)
    if study.startswith("passage-copy-"):
        from loopqwen.research.passage_copy import verify as verify_passage_copy

        verify_passage_copy(root, decision)
    if study.startswith("packed-release-retention-"):
        from loopqwen.research.packed_release import verify as verify_packed_release

        verify_packed_release(root, decision)
    if study.startswith("selective-raw-"):
        from loopqwen.research.selective_raw import verify as verify_selective_raw

        verify_selective_raw(root, decision)
    if study.startswith("raw-distillation-"):
        from loopqwen.research.raw_distillation import verify as verify_raw

        verify_raw(root, decision)
    if study.startswith("transport-lossless-"):
        from loopqwen.research.transport import verify as verify_transport

        verify_transport(root, decision)
    if study.startswith("narrative-full-book-"):
        from loopqwen.research.narrative_book import verify as verify_narrative

        verify_narrative(root, decision)
    if study.startswith("table-parser-"):
        from loopqwen.research.table_parser_study import verify as verify_table_parser

        verify_table_parser(root, decision)
    if study.startswith("operand-planner-"):
        from loopqwen.research.operand_study import verify as verify_operands

        verify_operands(root, decision)
    if study.startswith("conversation-followup-"):
        from loopqwen.research.conversation_study import verify as verify_conversation

        verify_conversation(root, decision)
    if study.startswith("release-mixed-field-"):
        from loopqwen.research.mixed_field import verify as verify_mixed_field

        verify_mixed_field(root, decision)
    if study.startswith("linked-reading-"):
        from loopqwen.research.linked_reading import verify as verify_linked_reading

        verify_linked_reading(root, decision)
    if study.startswith("bridge-reading-"):
        from loopqwen.research.bridge_reading import verify as verify_bridge_reading

        verify_bridge_reading(root, decision)
    if study.startswith("cache-language-"):
        from loopqwen.research.cache_language import verify as verify_cache_language

        verify_cache_language(root, decision)
    if study.startswith("tool-language-release-"):
        from scripts.tool_release import verify as verify_tool_release

        verify_tool_release(root, decision)
    if study.startswith(
        (
            "register-allocation-",
            "register-page-prefill-",
            "register-page-full-",
            "register-packed-",
            "register-fused-base-",
        )
    ):
        from loopqwen.research.register_allocation import verify as verify_allocation

        verify_allocation(root, decision)
    if study.startswith("register-rebalanced-"):
        from loopqwen.research.balanced_register import verify as verify_register

        verify_register(root, decision)
    if study.startswith("tool-prefix-"):
        from scripts.tool_prefix_study import verify as verify_tool_prefix

        verify_tool_prefix(root, decision)
    if study.startswith("contrast-boolq-"):
        from scripts.contrast_study import verify as verify_contrast

        verify_contrast(root, decision)
    if study.startswith("tool-bfcl-"):
        from scripts.tool_bfcl_study import verify as verify_external_calls

        verify_external_calls(root, decision)
    if study.startswith(("tool-dialogue-", "schema-calls-")):
        from scripts.tool_audit import verify as verify_tool_dialogue

        verify_tool_dialogue(root, decision, read, sha, require)
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
    if study in (
        "evidence-language-release-01",
        "native-language-release-01",
        "native-language-release-02",
    ):
        from scripts.research_depth_audit import verify_release

        verify_release(root, decision, read, sha, require)
    if study.startswith("native-scope-retention-"):
        from scripts.research_depth_audit import verify_scope

        verify_scope(root, decision, read, sha, require)
    if study.startswith("science-composition-"):
        from scripts.research_depth_audit import verify_science

        verify_science(root, decision, read, sha, require)
    if study == "short-depth-transfer-01" or study.startswith(
        ("deduction-", "native-induction-", "native-kinship-")
    ):
        from scripts import research_depth_audit

        research_depth_audit.verify(
            root, decision, read, sha, require, first_turn_answer
        )
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
    if study in ("context-128k-01", "context-128k-disk", "context-rationale-128k"):
        verify_context(root, decision)
    if study in (
        "attention-memory-128k",
        "conditional-depth-128k",
        "conditional-language-128k",
    ):
        verify_full_memory(root, decision)
    if study == "natural-context-01":
        verify_natural_context(root, decision)
    if study == "prose-reading-01":
        verify_prose_reading(root, decision)
    if study == "shared-budget-vote-01":
        verify_shared_budget(root, decision)
    if study == "attention-memory-01":
        verify_attention_memory(root, decision)
    if study in (
        "conditional-depth-01",
        "conditional-language-01",
        "conditional-language-02",
    ):
        verify_system(root, decision)
    if study in ("entailment-triplets-01", "entailment-triplets-confirm-01"):
        verify_entailment(root, decision)
    if study == "prefill-block-01":
        verify_prefill(root, decision)
    if study == "ledger-curriculum-01":
        verify_ledger_curriculum(root, decision)
    if study == "entailment-dense-01":
        verify_inference_attribution(root, decision)
    if study == "learned-exit-dense-01":
        verify_exit_attribution(root, decision)
    if study == "rationale-dense-01":
        verify_dense_rationale(root, decision)
    if study in (
        "rationale-transfer-01",
        "checked-rationale-01",
        "paired-solutions-01",
        "all-layer-reasoning-01",
        "recurrent-deliberation-01",
        "projected-distillation-01",
        "privileged-reasoning-01",
        "late-recurrent-exit-01",
        "learned-exit-01",
        "learned-exit-confirm-01",
        "fullrank-reasoning-01",
    ):
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


def first_turn_answer(tokens, text):
    # Atomic IDs from the pinned tokenizer. Score only a complete first assistant turn.
    if (
        not tokens
        or tokens[-1] not in (248044, 248046)
        or 248068 in tokens
        or tokens.count(248069) != 1
    ):
        return None
    require(
        not {248044, 248046}.intersection(tokens[:-1]), "Tokens after first boundary"
    )
    boundary = "<|im_end|>" if tokens[-1] == 248046 else "<|endoftext|>"
    require(
        text.endswith(boundary) and text.count("</think>") == 1,
        "Turn text/token mismatch",
    )
    return numeric(text.removesuffix(boundary).split("</think>")[1])


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
    if "base_archive" in reg:
        source = Path(reg["base_archive"])
        qualified = read(Path(reg["candidate"]).parent / "decision.json")
        require(
            read(source / "prompt.json.gz") == prompt
            and sha(source / "base.json")
            == sha(root / "base.json")
            == reg["base_row_sha256"]
            and read(source / "decision.json")["capacity"]
            and qualified["passed"]
            and qualified["weights"] == reg["weights"]
            and sha(Path(reg["candidate"]) / "adapters.safetensors") == reg["weights"],
            "Language deployment and measured Base identity",
        )
    require(
        sha(Path(reg["candidate"]) / "adapters.safetensors") == reg["weights"],
        "Registered context checkpoint",
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
            row["greedy"] and row["max_tokens"] == reg["generation_cap"],
            "Context generation policy",
        )
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
            row["answer"] == first_turn_answer(row["token_ids"], row["text"])
            and row["correct"] == (row["answer"] == prompt["gold"]),
            "Context retrieval score",
        )
        completed.append(arm)
    require(completed == decision["completed"], "Context arm inventory")
    require(
        decision["capacity"] == (completed == ["base", "candidate"]),
        "Context capacity gate",
    )
    if "resident_archive" in decision:
        resident = Path(decision["resident_archive"])
        for arm in completed:
            a, b = read(root / f"{arm}.json"), read(resident / f"{arm}.json")
            require(
                (a["input_sha256"], a["weights"]) == (b["input_sha256"], b["weights"])
                and decision["identical_generations"][arm]
                == (a["token_ids"] == b["token_ids"]),
                "Long decode parity",
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


def verify_retention_pair(first, second):
    manifest = read(Path("docs/measurements/phase4/v3_retention-manifest.json"))
    domains = set(manifest["sources"])
    counts = {k: r["tokens"] for k, r in first["by_domain"].items()}
    require(
        sum(counts.values()) == 9328 and set(counts) == domains, "Retention exposure"
    )
    for measurements in (first, second):
        groups = measurements["by_domain"]
        require(
            measurements["sha256"] == manifest["sha256"] and set(groups) == domains,
            "Retention sources",
        )
        for domain, row in groups.items():
            require(
                row["tokens"] == counts[domain]
                and row["tokens"] > 0
                and row["nll"] == row["loss"] / row["tokens"],
                "Domain retention loss",
            )
        require(
            measurements["nll"] == sum(r["loss"] for r in groups.values()) / 9328,
            "Pooled retention loss",
        )


def verify_dense_rationale(root, decision):
    reg = read(root / "registration.json")
    require(
        sha(root / "registration.json") == decision["registration"],
        "Dense registration",
    )
    if decision["status"] == "interrupted":
        require(
            not decision["passed"] and decision["reason"],
            "Interrupted dense comparison",
        )
        return
    prior = Path(reg["prerequisite"])
    earlier = read(prior / "decision.json")
    state = read(root / "model/state.json")
    dense = read(root / "dense.json.gz")
    candidate = read(prior / "candidate.json.gz")
    require(
        dense["weights"]
        == sha(root / "model/adapters.safetensors")
        == decision["weights"]
        and candidate["weights"] == reg["recurrent_weights"],
        "Dense pair weights",
    )
    require(
        state["parent"] == reg["parent_sha256"]
        and decision["updates"] == state["step"] == 32
        and decision["targets"]
        == sum(h["targets"] for h in state["history"])
        == earlier["targets"]
        and decision["data"] == earlier["data"],
        "Dense exposure",
    )
    diff = []
    for a, c in zip(dense["records"], candidate["records"], strict=True):
        require(
            (a["prompt_ids"], a["id"], a["gold"])
            == (c["prompt_ids"], c["id"], c["gold"]),
            "Unpaired dense prompts",
        )
        answer = numeric(a["visible"]) if a["complete"] else None
        require(
            answer == a["answer"] == first_turn_answer(a["token_ids"], a["text"])
            and a["correct"] == (answer == a["gold"]),
            "Dense answer",
        )
        diff.append(int(c["correct"]) - int(a["correct"]))
    d, r = decision["dense"], decision["recurrent"]
    verify_retention_pair(d["retention"], r["retention"])
    require(
        len(diff) == 32
        and r == earlier["candidate"]
        and d["correct"] == sum(x["correct"] for x in dense["records"])
        and d["seconds"] == sum(x["seconds"] for x in dense["records"])
        and (decision["wins"], decision["losses"]) == (diff.count(1), diff.count(-1)),
        "Dense aggregate",
    )
    require(
        decision["passed"]
        == (
            sum(diff) >= 2
            and r["seconds"] <= 1.25 * d["seconds"]
            and r["retention"]["nll"] <= d["retention"]["nll"] + 0.03
        ),
        "Dense gate",
    )


def verify_preference_training(root, reg, decision, items):
    pool_file = Path("research/data/paired-solutions-pool.json")
    pool = read(pool_file)
    fresh = {r["id"] for r in pool + items}
    exclusions = {
        r["id"]
        for name in ("rationale-transfer-01", "checked-rationale-01")
        for r in read(Path("research/data") / (name + ".json"))
    }
    exclusions.update(
        r["question_id"]
        for r in read(Path("research/data/checked-rationale-training.json"))["selected"]
    )
    require(
        sha(pool_file) == decision["pool"]
        and len(fresh) == len(pool) + len(items)
        and len(pool) == reg["pool"]
        and fresh.isdisjoint(exclusions),
        "Preference cohorts",
    )
    if not (root / "rollouts.json.gz").exists():
        require(
            decision["status"] == "interrupted" and decision["reason"],
            "Missing sampled solutions",
        )
        return
    draws = read(root / "rollouts.json.gz")
    selected = read(root / "pairs.json")
    recovered = []
    for i, step in enumerate(draws):
        attempts = step["attempts"]
        require(
            step["id"] == pool[i]["id"] and 0 < len(attempts) <= reg["draws"],
            "Rollout order",
        )
        choices = []
        for j, r in enumerate(attempts):
            require(not choices, "Sampling continued after the first eligible pair")
            require(
                r["prompt_ids"] == pool[i]["prompt_ids"]
                and 0 < len(r["token_ids"]) <= reg["rollout_cap"]
                and r["seconds"] > 0,
                "Sampled request",
            )
            ans = first_turn_answer(r["token_ids"], r["text"])
            require(
                r["answer"] == ans and r["correct"] == (ans == pool[i]["gold"]),
                "Sampled solution verdict",
            )
            for k in range(j):
                other = attempts[k]
                if (
                    ans is None
                    or other["answer"] is None
                    or r["correct"] == other["correct"]
                ):
                    continue
                lengths = (len(r["token_ids"]), len(other["token_ids"]))
                if max(lengths) <= 1.5 * min(lengths):
                    good, bad = (j, k) if r["correct"] else (k, j)
                    choices.append((max(lengths) - min(lengths), good, bad))
        if choices:
            _, good, bad = min(choices)
            recovered.append({"pool_index": i, "chosen": good, "rejected": bad})
        else:
            require(len(attempts) == reg["draws"], "Incomplete sampled question")
    require(
        selected == recovered and len(selected) <= reg["max_pairs"],
        "Preference pair selection",
    )
    if decision["status"] == "interrupted":
        return
    require(
        decision["eligible"] == len(selected)
        and decision["sampled_questions"] == len(draws)
        and (len(draws) == reg["pool"] or len(selected) == reg["max_pairs"])
        and decision["sampling_seconds"]
        == sum(v["seconds"] for item in draws for v in item["attempts"]),
        "Preference collection costs or stop",
    )
    if decision["status"] == "insufficient_pairs":
        require(
            len(selected) < reg["min_pairs"]
            and not decision["passed"]
            and not (root / "model").exists(),
            "Insufficient pair gate",
        )
        return
    n = (len(selected) // reg["logical_batch"]) * reg["logical_batch"]
    state = read(root / "model/state.json")
    refs = read(root / "reference-nll.json")
    require(
        n >= reg["min_pairs"]
        and len(refs) == n
        and all(
            len(pair) == 2 and all(math.isfinite(v) and v > 0 for v in pair)
            for pair in refs
        ),
        "Frozen reference probabilities",
    )
    require(
        sha(root / "model/adapters.safetensors") == decision["weights"]
        and state["parent"] == reg["parent_sha256"]
        and state["step"]
        == decision["updates"]
        == reg["epochs"] * n // reg["logical_batch"]
        and decision["update_seconds"] == sum(h["seconds"] for h in state["history"]),
        "Preference optimizer lineage",
    )
    for epoch in range(reg["epochs"]):
        used = [
            j for h in state["history"] if h["epoch"] == epoch for j in h["indices"]
        ]
        require(sorted(used) == list(range(n)), "Preference epoch exposure")


def verify_distillation(root, reg, decision, items):
    from loopqwen.identity import fingerprint

    manifest = Path(
        reg.get(
            "training_manifest", "research/data/projected-distillation-training.json"
        )
    )
    plan = read(manifest)
    require(
        sha(manifest) == decision["training"]
        and sha(Path(reg["parent"]) / "adapters.safetensors") == reg["parent_sha256"]
        and sha(Path(reg["dense_control"]) / "adapters.safetensors")
        == reg["dense_sha256"]
        and set(plan["ordered_questions"]).isdisjoint(r["id"] for r in items),
        "Distillation source material",
    )
    original = plan["exposure"]
    exposure = [0 if d["raw"] else d["length"] - d["prefix"] for d in original]
    if "hidden_weight" in reg:
        if decision["status"] == "interrupted" and "weights" not in decision:
            return
        aux = [(i, d) for i, d in enumerate(original) if "teacher_prefix" in d]
        with (root / "teachers.safetensors").open("rb") as handle:
            header = json.loads(handle.read(int.from_bytes(handle.read(8), "little")))
        expected = {f"{i}.hidden" for i, _ in aux}
        require(
            len(aux) == reg["updates"]
            and {k for k in header if k.endswith(".hidden")} == expected,
            "Privileged state inventory",
        )
        for i, d in aux:
            positions = min(reg["positions"], d["length"] - d["prefix"])
            require(
                d["teacher_length"] - d["teacher_prefix"] == d["length"] - d["prefix"]
                and d["teacher_prefix"] > d["prefix"]
                and header[f"{i}.hidden"]["shape"] == [positions, 1024]
                and header[f"{i}.hidden"]["dtype"] == "BF16",
                "Privileged target alignment",
            )
    else:
        eligible = [i for i in range(len(original)) if i % reg["group"] < 3]
        eligible.sort(key=lambda i: fingerprint([reg["seed"], "teacher", i]))
        path = root / "training-draws.json.gz"
        draws = read(path)["records"] if path.exists() else []
        good = []
        require(len(draws) <= reg["max_draws"], "Teacher draw bound")
        for index, r in zip(eligible, draws):
            d = original[index]
            require(
                index == r["index"]
                and fingerprint(r["prompt_ids"]) == d["prompt"]
                and d["gold"] == r["gold"]
                and len(r["token_ids"]) <= reg["draw_cap"]
                and r["answer"] == first_turn_answer(r["token_ids"], r["text"])
                and r["correct"] == (r["answer"] == r["gold"]),
                "Teacher trajectory",
            )
            if r["correct"]:
                good.append(r)
        if decision["status"] == "insufficient_correct_training_traces":
            require(
                len(draws) == reg["max_draws"]
                and len(good) < reg["min_good"]
                and not decision["passed"],
                "Incorrect teacher-yield decision",
            )
            return
        if decision["status"] == "interrupted" and "weights" not in decision:
            return
        require(
            reg["min_good"] <= len(good) <= reg["wanted"]
            and len(good) == decision["correct_training_draws"]
            and (len(good) == reg["wanted"] or len(draws) == reg["max_draws"]),
            "Selected teacher yield",
        )
        selection = read(root / "draw-selection.json")
        require(
            selection["indices"] == [r["index"] for r in good],
            "Selected trajectory identity",
        )
        positions = [i for i in eligible if i not in {r["index"] for r in draws}]
        replacement = []
        for r in good:
            exposure[r["index"]] = len(r["token_ids"])
            for _ in range(reg["copies"] - 1):
                i = positions.pop()
                exposure[i] = len(r["token_ids"])
                replacement.append([i, r["index"]])
        require(
            selection["additional_slots"] == replacement,
            "Unregistered training replication",
        )
    state = read(root / "model/state.json")
    teach = read(root / "teachers.json")
    projected = read(root / "projection.json")
    require(
        sha(root / "model/adapters.safetensors") == decision["weights"]
        and state["step"] == reg["updates"] == decision["updates"]
        and state["parent"] == reg["parent_sha256"]
        and sum(exposure)
        == decision["targets"]
        == sum(h["targets"] for h in state["history"])
        and sum(h["seconds"] for h in state["history"]) == decision["update_seconds"]
        and sum(h["teacher_positions"] for h in state["history"]) == teach["positions"]
        and teach["rows"] == reg["updates"] * reg["group"]
        and sha(root / "teachers.safetensors")
        == teach["sha256"]
        == decision["teacher_targets"]
        and state["spec"]["deployment"]["config"]["rank"] == reg["rank"] + 8
        and len({r["projection"] for r in projected}) == len(projected) > 100
        and all(0 <= r["fraction"] <= 1.0001 for r in projected),
        "Projected adapter, teacher targets or optimizer exposure",
    )


def verify_system(root, decision):
    from loopqwen import REVISION
    from loopqwen.evaluation.mixed import assess, summarize
    from loopqwen.identity import fingerprint

    reg = read(root / "registration.json")
    path = Path("research/data") / (reg["id"] + ".json")
    train = path.with_name(reg["id"] + "-routing.json")
    data = read(path)
    manifest = read(train)
    require(
        sha(path) == decision["data"]
        and sha(train) == decision["training"]
        and sha(root / "registration.json") == decision["registration"],
        "Conditional comparison identity",
    )
    if decision["status"] == "incomplete_frozen_cohort":
        require(
            not decision["passed"]
            and decision["reason"]
            and not any(
                (root / p).exists()
                for p in ("candidate.json.gz", "base.json.gz", "model")
            ),
            "Closed before native measurement",
        )
        families = {
            k: sum(r["family"] == k for r in data) for k in {r["family"] for r in data}
        }
        require(
            families == decision["counts"]
            and families["reading"] < 8
            and len(data) < 44 + 6 * reg["relation_groups_per_source"],
            "Registered reading cohort incomplete",
        )
        return
    require(
        sum(r["family"] == "reading" for r in data) == 8
        and len({r["source"] for r in data if r["family"] == "reading"}) == 8,
        "Eight distinct reading articles",
    )
    require(
        len({r["id"] for r in data})
        == len(data)
        == 44 + 6 * reg.get("relation_groups_per_source", 2)
        and len(manifest["ids"]) == 4 * reg["route_examples_per_class"]
        and len(set(manifest["ids"])) == len(manifest["ids"])
        and set(manifest["ids"]).isdisjoint(fingerprint(r["prompt_ids"]) for r in data),
        "Independent mixed requests",
    )
    require(
        sha(Path(reg["reader"]).parent / "checksums.json")
        == reg["reader_qualification"]
        and sha(Path(reg["dense_control"]) / "adapters.safetensors")
        == reg["dense_sha256"],
        "Retained reader and dense control",
    )
    if "weights" in decision:
        folder = root / "model"
        meta = read(folder / "context-policy.json")
        report = read(root / "router-training.json")
        require(
            sha(folder / "context-policy.json") == decision["weights"]
            and meta["schema"] == 2
            and meta["confidence"] == reg["confidence"]
            and set(meta["experts"]) == set(reg["experts"])
            and report["documents"] == manifest["documents"]
            and report["n"] == len(manifest["ids"])
            and len(report["history"]) == reg["route_updates"]
            and all(math.isfinite(x) for x in report["history"]),
            "Selector and source identity",
        )
        if "selector_source" in reg:
            source = reg["selector_source"]
            arc = Path(source["directory"])
            require(
                report["source"] == source
                and sha(arc / "checksums.json") == source["inventory"]
                and sha(arc / "model/selector.safetensors")
                == sha(folder / "selector.safetensors")
                == source["sha256"],
                "Unchanged input selector",
            )
        for f, h in meta["files"].items():
            require(sha(folder / f) == h, "System tensor or policy digest")
        for name, info in reg["experts"].items():
            require(
                sha(folder / name / "adapters.safetensors")
                == info["weights"]
                == sha(Path(info["model"]) / "adapters.safetensors"),
                "Trained specialist identity",
            )
            for key in ("qualification", "confirmation"):
                stage = info[key]
                arc = Path(stage["directory"])
                require(
                    sha(arc / "checksums.json") == stage["inventory"]
                    and read(arc / "decision.json")["passed"]
                    and sha(Path(stage["audit"])) == stage["audit_sha256"]
                    and read(arc / "decision.json")["weights"] == info["weights"],
                    "Qualified specialist required",
                )
    reports = {}
    for arm in ("candidate", "base"):
        f = root / (arm + ".json.gz")
        if not f.exists():
            require(decision["status"] == "interrupted", "Missing system generations")
            continue
        obj = read(f)
        require(
            obj["weights"] == (REVISION if arm == "base" else decision["weights"])
            and len(obj["records"]) <= len(data),
            "System decoder identity",
        )
        reports[arm] = obj["records"]
        for r, q in zip(reports[arm], data):
            require(
                r["id"] == q["id"]
                and r["input_sha256"] == fingerprint(q["prompt_ids"])
                and r["prompt_tokens"] == len(q["prompt_ids"])
                and r["max_tokens"] == q["cap"]
                and r["greedy"]
                and 0 < len(r["token_ids"]) <= q["cap"]
                and r["elapsed_s"] >= r["ttft_s"] > 0,
                "System native generation and token budget",
            )
            expected = assess(q, r["token_ids"], r["text"])
            require(all(r[k] == v for k, v in expected.items()), "System answer")
    if decision["status"] == "interrupted":
        require(
            not decision["passed"] and decision["reason"], "Conditional worker stop"
        )
        return
    require(
        all(len(reports[a]) == len(data) for a in ("base", "candidate")),
        "Complete mixed pair",
    )
    scores = summarize(data, reports)
    a = scores["all"]
    kept = all(
        c["token_ids"] == b["token_ids"]
        for c, b in zip(reports["candidate"], reports["base"], strict=True)
        if c["context_route"] == "ordinary"
    )
    raw_rows = Path("data/v3_retention/validation.jsonl")
    require(
        decision["raw_route_ids"]
        == [json.loads(line)["id"] for line in raw_rows.read_text().splitlines()],
        "Ordinary routing of every raw retention passage",
    )
    ret = decision["retention"]
    verify_retention_pair(ret["base"], ret["candidate"])
    passed = (
        a["wins"] - a["losses"] >= reg["net_floor"]
        and all(v["wins"] >= v["losses"] for v in scores.values())
        and sum(v["wins"] > v["losses"] for k, v in scores.items() if k != "all") >= 2
        and sum(
            scores[k]["wins"] - scores[k]["losses"]
            for k in ("arithmetic", "entailment")
        )
        > 0
        and all(
            scores[k]["wins"] > scores[k]["losses"]
            for k in reg.get("required_gains", [])
        )
        and scores["register"]["candidate"]["correct"] == 4
        and a["candidate"]["seconds"] <= a["base"]["seconds"]
        and kept
        and ret["base"] == ret["candidate"]
    )
    require(
        decision["passed"] == passed
        and decision["scores"] == scores
        and decision["ordinary_identity"] == kept,
        "System retention and promotion rule",
    )


def verify_exit_attribution(root, decision):
    from loopqwen.evaluation.statistics import exact_mcnemar
    from loopqwen.identity import fingerprint

    s = read(root / "registration.json")
    info = s["expert"]
    rows, previous = [], []
    for stage in [info["qualification"], info["confirmation"]]:
        arc = Path(stage["directory"])
        require(
            sha(arc / "checksums.json") == stage["inventory"]
            and read(arc / "decision.json")["passed"],
            "Qualified original pair",
        )
        require(
            sha(Path(stage["audit"])) == stage["audit_sha256"], "Audited original pair"
        )
        rows += read(Path("research/data") / (arc.name + ".json"))
        previous += read(arc / "candidate.json.gz")["records"]
    require(
        sha(root / "registration.json") == decision["registration"]
        and fingerprint(rows) == decision["data"]
        and sha(Path(s["initial_dense"]) / "adapters.safetensors") == s["dense_sha256"],
        "Attribution inputs",
    )
    original = Path(info["qualification"]["directory"])
    first = read(original / "decision.json")
    if "dense_weights" in decision:
        folder = root / "model"
        ds = read(folder / "state.json")
        config = read(folder / "deliberation.json")
        fit = read(root / "readout-fit.json")
        ref_fit = read(original / "readout-fit.json")
        require(
            ds["step"] == decision["updates"] == first["updates"]
            and ds["parent"] == s["dense_sha256"]
            and ds["spec"]["depth"] == config["finisher_depth"] == 1
            and ds["spec"]["deployment"]["config"]["passes"] == 1
            and decision["targets"]
            == first["targets"]
            == sum(r["targets"] for r in ds["history"])
            and decision["documents"] == first["documents"]
            and all(
                fit[k] == ref_fit[k]
                for k in ("features", "updates", "positives", "negatives")
            )
            and config["switch_after"]
            == read(original / "model/deliberation.json")["switch_after"]
            and config["adapter_sha256"]
            == decision["dense_weights"]
            == sha(folder / "adapters.safetensors")
            and config["readout_sha256"] == sha(folder / "exit-readout.safetensors"),
            "Dense exposure, readout and checkpoint",
        )
    results = {}
    for name in ("dense", "recurrent"):
        f = root / (name + ".json.gz")
        if not f.exists():
            require(decision["status"] == "interrupted", "Missing attribution arm")
            continue
        pack = read(f)
        require(
            pack["weights"]
            == (decision["dense_weights"] if name == "dense" else info["weights"]),
            "Attribution weights",
        )
        runs = pack["records"]
        require(len(runs) <= len(rows), "Attribution extra requests")
        results[name] = runs
        for item, v, prior in zip(rows, runs, previous):
            ids = v["token_ids"]
            close = ids.index(248069) if 248069 in ids else None
            ended = ids[-1] in (248044, 248046)
            complete = ended and ids.count(248069) == 1 and 248068 not in ids
            visible = numeric(v["visible"]) if complete else None
            require(
                v["id"] == item["id"]
                and v["prompt_ids"] == item["prompt_ids"]
                and v["gold"] == item["gold"]
                and v["complete"] == complete
                and v["close_at"] == close
                and v["answer"] == visible
                and v["correct"] == (visible == item["gold"])
                and len(ids) <= 1024
                and not set(ids[:-1]).intersection((248044, 248046))
                and v["seconds"] >= v["ttft"] > 0
                and ids[:704] == prior["token_ids"][:704],
                "Matched first-turn attribution",
            )
            if name == "recurrent":
                require(ids == prior["token_ids"], "Fresh unchanged recurrent stream")
    if decision["status"] == "interrupted":
        require(not decision["passed"] and decision["reason"], "Attribution stop")
        return
    require(
        all(len(rs) == len(rows) for rs in results.values()),
        "Complete attribution pair",
    )
    scores = {}
    for source in ("all", *sorted({q["source"] for q in rows})):
        idx = [
            i for i, r in enumerate(rows) if source == "all" or r["source"] == source
        ]
        summary = {}
        for arm, runs in results.items():
            val = [runs[i] for i in idx]
            summary[arm] = {
                "correct": sum(r["correct"] for r in val),
                "valid": sum(r["answer"] is not None for r in val),
                "seconds": sum(r["seconds"] for r in val),
                "tokens": sum(len(r["token_ids"]) for r in val),
            }
        delta = [
            int(results["recurrent"][i]["correct"])
            - int(results["dense"][i]["correct"])
            for i in idx
        ]
        w, l = delta.count(1), delta.count(-1)
        scores[source] = dict(
            **summary, n=len(idx), wins=w, losses=l, p=exact_mcnemar(w, l)
        )
    a = scores["all"]
    ret = decision["retention"]
    verify_retention_pair(ret["dense"], ret["recurrent"])
    passed = (
        a["wins"] - a["losses"] >= s["net_floor"]
        and all(r["wins"] >= r["losses"] for r in scores.values())
        and a["recurrent"]["seconds"] <= s["max_time_ratio"] * a["dense"]["seconds"]
        and ret["recurrent"]["nll"] <= ret["dense"]["nll"] + 0.03
    )
    require(
        scores == decision["scores"] and passed == decision["passed"],
        "Dense attribution recount",
    )


def verify_inference_attribution(root, decision):
    from loopqwen.evaluation.relations import conclusion, summarize

    s = read(root / "registration.json")
    expert = s["expert"]
    manifest = Path("research/data/entailment-triplets-training.json")
    require(
        sha(root / "registration.json") == decision["registration"]
        and sha(manifest) == decision["training"]
        and sha(Path(s["dense_control"]) / "adapters.safetensors") == s["dense_sha256"],
        "NLI attribution identity",
    )
    items, ref, hashes = [], [], []
    for name in ("qualification", "confirmation"):
        info = expert[name]
        arc = Path(info["directory"])
        require(
            sha(arc / "checksums.json") == info["inventory"]
            and sha(Path(info["audit"])) == info["audit_sha256"]
            and read(arc / "decision.json")["passed"],
            "NLI attribution prerequisites",
        )
        path = Path("research/data") / (arc.name + ".json")
        items += read(path)
        hashes.append(sha(path))
        ref += read(arc / "candidate.json.gz")["records"]
    require(
        hashes == decision["panels"]
        and len({v["id"] for v in items}) == len(items) == 120,
        "NLI attribution cohort",
    )
    if "weights" in decision:
        ds = read(root / "model/state.json")
        baseline = read(Path(expert["qualification"]["directory"]) / "decision.json")
        require(
            sha(root / "model/adapters.safetensors") == decision["weights"]
            and ds["parent"] == s["dense_sha256"]
            and ds["spec"]["deployment"]["config"]["passes"] == 1
            and ds["step"] == decision["updates"] == baseline["updates"]
            and sum(h["targets"] for h in ds["history"])
            == decision["targets"]
            == read(manifest)["response_tokens"]
            == baseline["targets"],
            "Matched dense training",
        )
    data = {}
    for a in ("dense", "recurrent"):
        p = root / (a + ".json.gz")
        if not p.exists():
            require(
                decision["status"] == "interrupted",
                "Missing NLI attribution generations",
            )
            continue
        pkg = read(p)
        require(
            pkg["weights"]
            == (decision["weights"] if a == "dense" else expert["weights"]),
            "NLI attribution weights",
        )
        rows = pkg["records"]
        data[a] = rows
        require(len(rows) <= len(items), "Extra paired generations")
        for r, item, prior in zip(rows, items, ref):
            ids = r["token_ids"]
            complete = (
                ids[-1] in (248044, 248046)
                and ids.count(248069) == 1
                and 248068 not in ids
            )
            logits = r["probe_logits"]
            require(
                r["id"] == item["id"]
                and r["prompt_ids"] == item["prompt_ids"]
                and 0 < len(ids) <= 1024
                and not set(ids[:-1]).intersection((248044, 248046))
                and r["seconds"] >= r["ttft"] > 0
                and r["complete"] == complete
                and r["answer"] == (conclusion(r["visible"]) if complete else None)
                and r["correct"] == (r["answer"] == item["label"])
                and len(logits) == 3
                and all(map(math.isfinite, logits))
                and r["probe"]
                == ("yes", "no", "maybe")[max(range(3), key=logits.__getitem__)],
                "NLI attribution first turn and probe",
            )
            if a == "recurrent":
                require(
                    all(
                        r[k] == prior[k]
                        for k in ("id", "prompt_ids", "token_ids", "probe_logits")
                    ),
                    "Frozen recurrent stream and logits",
                )
    if decision["status"] == "interrupted":
        require(
            not decision["passed"] and decision["reason"], "Attribution resource stop"
        )
        return
    require(all(len(v) == len(items) for v in data.values()), "Full NLI attribution")
    scores = summarize(items, {"base": data["dense"], "candidate": data["recurrent"]})
    for summary in scores.values():
        summary["dense"] = summary.pop("base")
        summary["recurrent"] = summary.pop("candidate")
    a = scores["all"]
    d, c = a["dense"], a["recurrent"]
    ret = decision["retention"]
    verify_retention_pair(ret["dense"], ret["recurrent"])
    gate = (
        a["wins"] - a["losses"] >= s["net_floor"]
        and all(scores[k]["wins"] >= scores[k]["losses"] for k in ("esnli", "multinli"))
        and c["probe"] > d["probe"]
        and c["triplets"] > d["triplets"]
        and c["seconds"] <= s["max_time_ratio"] * d["seconds"]
        and ret["recurrent"]["nll"] <= ret["dense"]["nll"] + 0.03
    )
    require(
        scores == decision["scores"] and gate == decision["passed"],
        "Verified NLI attribution",
    )


def verify_entailment(root, decision):
    from loopqwen import REVISION
    from loopqwen.evaluation.relations import conclusion, summarize

    spec = read(root / "registration.json")
    panel = Path("research/data") / (spec["id"] + ".json")
    manifest = Path("research/data/entailment-triplets-training.json")
    rows = read(panel)
    require(
        sha(root / "registration.json") == decision["registration"]
        and sha(panel) == decision["data"]
        and sha(manifest) == decision["training"],
        "Contrastive panel and training identity",
    )
    require(
        len(rows) == 6 * spec["per_source_groups"]
        and len({r["id"] for r in rows}) == len(rows),
        "Unique NLI items",
    )
    keys = read(manifest)["ordered_questions"]
    require(
        len(keys) == len(set(keys)) == 6 * spec["updates"]
        and set(keys).isdisjoint(r["id"] for r in rows),
        "NLI training separation",
    )
    require(
        len({r["premise"].casefold() for r in rows}) == len(rows) // 3,
        "Unique contrast groups",
    )
    for i in range(0, len(rows), 3):
        group = rows[i : i + 3]
        require(
            len({r["premise"] for r in group}) == 1
            and len({r["source"] for r in group}) == 1
            and {r["label"] for r in group} == {"yes", "no", "maybe"},
            "Three counterfactual relations",
        )
    require(
        sha(Path(spec["parent"]) / "adapters.safetensors") == spec["parent_sha256"]
        and sha(Path(spec["dense_control"]) / "adapters.safetensors")
        == spec["dense_sha256"],
        "NLI control integrity",
    )
    if "qualification" in spec:
        primary = Path(spec["model"]).parent
        earlier = read(panel.with_name(primary.name + ".json"))
        require(
            sha(primary / "checksums.json")
            == spec["qualification"]
            == decision["qualification"]
            and read(primary / "decision.json")["passed"]
            and sha(Path(spec["model"]) / "adapters.safetensors")
            == decision["weights"]
            == spec["weights"]
            and sha(Path(spec["audit"])) == spec["audit_sha256"]
            and {r["premise"].casefold() for r in rows}.isdisjoint(
                r["premise"].casefold() for r in earlier
            ),
            "Frozen independent NLI confirmation",
        )
    elif "weights" in decision:
        state = read(root / "model/state.json")
        require(
            sha(root / "model/adapters.safetensors") == decision["weights"]
            and state["parent"] == spec["parent_sha256"]
            and state["step"] == spec["updates"] == decision["updates"]
            and sum(h["targets"] for h in state["history"])
            == decision["targets"]
            == read(manifest)["response_tokens"],
            "NLI training exposure",
        )
    reports = {}
    for arm in ("candidate", "base"):
        path = root / (arm + ".json.gz")
        if not path.exists():
            require(decision["status"] == "interrupted", "Missing NLI generations")
            continue
        report = read(path)
        require(
            report["weights"] == (REVISION if arm == "base" else decision["weights"]),
            "NLI decoder weights",
        )
        records = report["records"]
        require(len(records) <= len(rows), "Extra generations")
        for row, item in zip(records, rows):
            ids = row["token_ids"]
            require(
                row["id"] == item["id"]
                and row["prompt_ids"] == item["prompt_ids"]
                and 0 < len(ids) <= spec["generation_cap"]
                and row["seconds"] >= row["ttft"] > 0,
                "NLI prompt, output and time",
            )
            complete = (
                ids[-1] in (248044, 248046)
                and 248068 not in ids
                and ids.count(248069) == 1
            )
            require(
                row["complete"] == complete
                and not set(ids[:-1]).intersection((248044, 248046)),
                "NLI assistant boundary",
            )
            if complete:
                boundary = "<|im_end|>" if ids[-1] == 248046 else "<|endoftext|>"
                require(
                    row["text"].endswith(boundary)
                    and row["text"].count("</think>") == 1
                    and row["visible"]
                    == row["text"].removesuffix(boundary).split("</think>")[1],
                    "NLI visible answer",
                )
            answer = conclusion(row["visible"]) if complete else None
            logits = row["probe_logits"]
            require(
                row["answer"] == answer
                and row["correct"] == (answer == item["label"])
                and len(logits) == 3
                and all(map(math.isfinite, logits))
                and row["probe_seconds"] > 0
                and row["probe"]
                == ("yes", "no", "maybe")[max(range(3), key=logits.__getitem__)],
                "NLI visible and latent decisions",
            )
        reports[arm] = records
    if decision["status"] == "interrupted":
        require(not decision["passed"] and decision["reason"], "NLI stop")
        return
    require(
        all(len(reports[a]) == len(rows) for a in ("base", "candidate")),
        "Incomplete NLI pair",
    )
    scores = summarize(rows, reports)
    ret = decision["retention"]
    verify_retention_pair(ret["base"], ret["candidate"])
    a = scores["all"]
    qualify = (
        a["wins"] - a["losses"] >= spec["net_floor"]
        and all(scores[k]["wins"] > scores[k]["losses"] for k in ("esnli", "multinli"))
        and a["candidate"]["seconds"] <= a["base"]["seconds"]
        and a["candidate"]["probe"] > a["base"]["probe"]
        and a["candidate"]["triplets"] > a["base"]["triplets"]
        and ret["candidate"]["nll"] <= ret["base"]["nll"] + 0.03
    )
    require(
        scores == decision["scores"] and decision["passed"] == qualify,
        "Reconstructed inference gate",
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
    trained = reg["id"] != "rationale-transfer-01"
    if reg["id"] == "paired-solutions-01":
        verify_preference_training(root, reg, decision, items)
        if decision["status"] == "insufficient_pairs":
            return
    if reg["id"] in ("projected-distillation-01", "privileged-reasoning-01"):
        verify_distillation(root, reg, decision, items)
        if decision["status"] == "insufficient_correct_training_traces":
            return
    weight = decision.get("weights") if trained else reg["weights"]
    if reg["id"] == "checked-rationale-01":
        manifest = Path("research/data/checked-rationale-training.json")
        training = read(manifest)
        question_ids = {row["question_id"] for row in training["selected"]}
        prior_panel = read(Path("research/data/rationale-transfer-01.json"))
        require(
            sha(manifest) == decision["training"]
            and len(question_ids)
            == len(training["selected"])
            == reg["updates"] * reg["logical_batch"]
            and question_ids.isdisjoint({r["id"] for r in items + prior_panel})
            and {r["id"] for r in items}.isdisjoint({r["id"] for r in prior_panel}),
            "Training and evaluation identities",
        )
        if weight:
            state = read(root / "model/state.json")
            require(
                weight == sha(root / "model/adapters.safetensors")
                and state["parent"] == reg["parent_sha256"]
                and state["step"] == decision["updates"] == reg["updates"]
                and decision["targets"] == sum(h["targets"] for h in state["history"]),
                "Checked supervision checkpoint",
            )
    if reg["id"] == "all-layer-reasoning-01":
        manifest = Path("research/data/all-layer-reasoning-training.json")
        schedule = read(manifest)
        keys = schedule["ordered_questions"]
        require(
            sha(manifest) == decision["training"]
            and len(set(keys)) == len(keys) == reg["updates"] * reg["logical_batch"]
            and set(keys).isdisjoint(r["id"] for r in items)
            and sha(Path(reg["parent"]) / "adapters.safetensors")
            == reg["parent_sha256"]
            and sha(Path(reg["dense_control"]) / "adapters.safetensors")
            == reg["dense_sha256"],
            "Expanded-adapter training cohort and controls",
        )
        if weight:
            st = read(root / "model/state.json")
            require(
                sha(root / "model/adapters.safetensors") == weight
                and st["parent"] == reg["parent_sha256"]
                and st["step"] == decision["updates"] == reg["updates"]
                and sum(h["targets"] for h in st["history"])
                == schedule["response_tokens"]
                == decision["targets"]
                and sum(h["seconds"] for h in st["history"])
                == decision["update_seconds"]
                and st["spec"]["deployment"]["config"]["target"] == "all"
                and st["spec"]["deployment"]["config"]["passes"] == 2,
                "Expanded-adapter checkpoint and exposure",
            )
    if reg["id"] == "recurrent-deliberation-01":
        manifest = Path("research/data/recurrent-deliberation-training.json")
        pool = read(manifest)
        require(
            sha(manifest) == decision["training"]
            and len({r["id"] for r in pool}) == len(pool) == reg["pool"]
            and {r["id"] for r in pool}.isdisjoint(r["id"] for r in items)
            and sha(Path(reg["parent"]) / "adapters.safetensors")
            == reg["parent_sha256"]
            and sha(Path(reg["dense_control"]) / "adapters.safetensors")
            == reg["dense_sha256"],
            "Draft training items and frozen controls",
        )
        drafts_file = root / "drafts.json.gz"
        drafts = read(drafts_file) if drafts_file.is_file() else []
        require(len(drafts) <= len(pool), "Extra training draft")
        for d, row in zip(drafts, pool):
            tokens = d["token_ids"]
            close = next((i for i, x in enumerate(tokens) if x == 248069), None)
            keep = (
                close // 2
                if close is not None
                else (len(tokens) if d["stop"] == "cap" else (len(tokens) - 1) // 2)
            )
            prefix = row["prompt_ids"] + tokens[:keep]
            doc = d["document"]
            require(
                d["id"] == row["id"]
                and d["prompt_ids"] == row["prompt_ids"]
                and 0 < len(tokens) <= reg["training_draft_cap"]
                and d["kept"] == keep
                and doc["ids"][: doc["prefix"]] == prefix
                and doc["ids"][-1] == 248046
                and doc["ids"][doc["prefix"] :].count(248069) == 1,
                "Invalid training prefix or completion",
            )
        if weight:
            st = read(root / "model/state.json")
            policy = read(root / "model/deliberation.json")
            require(
                len(drafts) == len(pool)
                and sha(root / "model/adapters.safetensors")
                == policy["adapter_sha256"]
                == weight
                and policy["revision"] == REVISION
                and policy["version"] == 1
                and policy["switch_after"] == reg["switch_after"]
                and st["parent"] == reg["parent_sha256"]
                and st["step"] == decision["updates"] == reg["updates"]
                and decision["targets"]
                == sum(h["targets"] for h in st["history"])
                == 2
                * sum(
                    len(d["document"]["ids"]) - d["document"]["prefix"] for d in drafts
                ),
                "Trained deliberation model or exposure",
            )
    if reg["id"] == "learned-exit-confirm-01":
        parent = Path(reg["model"]).parent
        first = read(parent / "decision.json")
        require(
            first["passed"]
            and sha(parent / "checksums.json")
            == reg["qualification"]
            == decision["qualification"]
            and weight
            == reg["weights"]
            == first["weights"]
            == sha(parent / "model/adapters.safetensors")
            and sha(Path(reg["dense_control"]) / "adapters.safetensors")
            == reg["dense_sha256"]
            and read(parent / "model/deliberation.json")["switch_after"]
            == reg["switch_after"]
            and {r["id"] for r in items}.isdisjoint(
                r["id"] for p in reg["excluded_panels"] for r in read(Path(p))
            )
            and {r["id"] for r in items}.isdisjoint(
                q
                for p in reg["excluded_training"]
                for q in read(Path(p))["ordered_questions"]
            ),
            "Fixed qualified checkpoint and independent confirmation items",
        )
    if reg["id"] == "fullrank-reasoning-01":
        manifest = Path("research/data/fullrank-reasoning-training.json")
        require(
            sha(manifest) == decision["training"]
            and set(read(manifest)["ordered_questions"]).isdisjoint(
                r["id"] for r in items
            )
            and sha(Path(reg["parent"]) / "adapters.safetensors")
            == reg["parent_sha256"]
            and sha(Path(reg["dense_control"]) / "adapters.safetensors")
            == reg["dense_sha256"],
            "Full-rank initialization, training and control identity",
        )
        calpath = root / "calibration.json"
        if calpath.exists():
            cal = read(calpath)
            table = cal["rows"]
            eligible = [
                r
                for r in table
                if r["nll"] <= cal["base"] + reg["calibration_tolerance"]
            ]
            chosen = max(eligible, key=lambda r: r["fraction"]) if eligible else None
            require(
                [r["fraction"] for r in table] == reg["fractions"]
                and all(r["tensors"] > 186 and math.isfinite(r["nll"]) for r in table)
                and cal["selected"] == chosen,
                "Training-only mixture selection",
            )
            if decision["status"] == "no_retained_mixture":
                require(
                    chosen is None
                    and not weight
                    and not decision["passed"]
                    and not (root / "candidate.json.gz").exists(),
                    "Calibration prerequisite",
                )
                return
            if weight:
                st = read(root / "model/state.json")
                mix = st["spec"]["deployment"]["backbone_mix"]
                from loopqwen.identity import fingerprint

                require(
                    mix == decision["backbone_mix"]
                    and mix["donor"] == reg["teacher"]
                    and mix["fraction"] == chosen["fraction"]
                    and mix["tensors"] == chosen["tensors"]
                    and st["parent"] == fingerprint([reg["base_revision"], mix])
                    and st["step"] == reg["updates"] == decision["updates"]
                    and sum(h["targets"] for h in st["history"])
                    == decision["targets"]
                    == read(manifest)["response_tokens"]
                    and sha(root / "model/adapters.safetensors") == weight,
                    "Materialized full-rank decoder and native optimization",
                )
        else:
            require(
                decision["status"] == "interrupted" and not weight,
                "Missing training calibration",
            )
    if reg["id"] == "learned-exit-01":
        manifest = Path("research/data/learned-exit-training.json")
        ids = read(manifest)["ordered_questions"]
        originals = read(Path(reg["drafts"]))
        require(
            sha(manifest) == decision["training"]
            and sha(Path(reg["drafts"])) == reg["draft_sha256"]
            and sha(Path(reg["parent"]) / "adapters.safetensors")
            == reg["parent_sha256"]
            and sha(Path(reg["dense_control"]) / "adapters.safetensors")
            == reg["dense_sha256"]
            and len(set(ids)) == len(ids) == len(originals) == 64
            and ids == [r["id"] for r in originals]
            and set(ids).isdisjoint(r["id"] for r in items),
            "Learned exit training ancestry and held-out material",
        )
        path = root / "extensions.json.gz"
        ext = read(path) if path.exists() else []
        require([v["id"] for v in ext] == ids[: len(ext)], "Base continuation order")
        for old, new in zip(originals, ext):
            active = old["close_at"] is None and old["stop"] == "cap"
            require(
                bool(new["token_ids"]) == active
                and (new["seconds"] > 0) == active
                and len(new["token_ids"]) + len(old["token_ids"])
                <= reg["training_draft_cap"],
                "Authentic extended thought",
            )
        if weight:
            state = read(root / "model/state.json")
            config = read(root / "model/deliberation.json")
            fit = read(root / "readout-fit.json")
            require(
                len(ext) == len(ids)
                and state["step"] == decision["updates"] == reg["updates"]
                and state["parent"] == reg["parent_sha256"]
                and sum(h["targets"] for h in state["history"]) == decision["targets"]
                and weight
                == config["adapter_sha256"]
                == sha(root / "model/adapters.safetensors")
                and decision["readout"]
                == config["readout_sha256"]
                == sha(root / "model/exit-readout.safetensors")
                and config["revision"] == REVISION
                and config["version"] == 2
                and config["switch_after"] == reg["switch_after"]
                and fit["updates"] == reg["head_updates"]
                and fit["positives"] >= len(ids)
                and fit["negatives"] >= len(ids)
                and fit["after"] < fit["before"],
                "Learned exit adapter, native vocabulary readout and objective",
            )
    if reg["id"] == "late-recurrent-exit-01":
        from loopqwen.identity import fingerprint

        train = Path("research/data/late-recurrent-exit-training.json")
        manifest = read(train)
        draft = Path(reg["drafts"])
        require(
            sha(train) == decision["training"]
            and sha(draft) == reg["draft_sha256"]
            and sha(Path(reg["parent"]) / "adapters.safetensors")
            == reg["parent_sha256"]
            and sha(Path(reg["dense_control"]) / "adapters.safetensors")
            == reg["dense_sha256"],
            "Exit training material",
        )
        samples = read(draft)
        options = [
            sorted(
                (r for r in samples if (r["close_at"] is None) == flag),
                key=lambda r: fingerprint([reg["seed"], r["id"]]),
            )[:16]
            for flag in (True, False)
        ]
        keys = [r["id"] for pair in zip(*options, strict=True) for r in pair]
        require(
            keys == manifest["ordered_questions"]
            and len(set(keys)) == 32
            and set(keys).isdisjoint(r["id"] for r in items),
            "Exit draft selection",
        )
        if weight:
            st = read(root / "model/state.json")
            policy = read(root / "model/deliberation.json")
            require(
                st["step"] == decision["updates"] == reg["updates"]
                and st["parent"] == reg["parent_sha256"]
                and sum(h["targets"] for h in st["history"])
                == manifest["targets"]
                == decision["targets"]
                and sha(root / "model/adapters.safetensors")
                == policy["adapter_sha256"]
                == weight
                and policy["revision"] == REVISION
                and policy["switch_after"] == reg["switch_after"],
                "Exit training and inference checkpoint",
            )
    for arm, expected_weight in (("base", REVISION), ("candidate", weight)):
        file = root / (arm + ".json.gz")
        if not file.exists():
            require(decision["status"] == "interrupted", "Missing transfer generations")
            continue
        report = read(file)
        require(report["weights"] == expected_weight, "Transfer weights")
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
                row["answer"] == answer == first_turn_answer(tokens, row["text"])
                and row["correct"] == (answer == item["gold"]),
                "Transfer numeric score",
            )
            require(
                0 < row["seconds_at_256"] <= row["seconds"], "Invalid prefix timing"
            )
        reports[arm] = rows
    if reg["id"] in (
        "recurrent-deliberation-01",
        "late-recurrent-exit-01",
        "learned-exit-01",
        "learned-exit-confirm-01",
    ):
        switch = reg["switch_after"]
        for a, c in zip(reports.get("base", ()), reports.get("candidate", ())):
            early = a["token_ids"][:switch]
            late = len(a["token_ids"]) > switch and 248069 not in early
            require(
                a["recurrent_from"] is None
                and c["recurrent_from"] == (switch if late else None)
                and c["token_ids"][:switch] == early
                and (late or a["token_ids"] == c["token_ids"]),
                "The early ordinary-depth trajectory differs from the pinned control",
            )
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
    if trained:
        r = decision["retention"]
        verify_retention_pair(r["base"], r["candidate"])
        short = decision["prefix_256"]
        passed = (
            full["all"]["wins"] - full["all"]["losses"] >= 2
            and full["all"]["p"] < 0.05
            and full["all"]["candidate"]["seconds"] <= full["all"]["base"]["seconds"]
            and all(
                full[src]["wins"] >= full[src]["losses"]
                and short[src]["wins"] > short[src]["losses"]
                for src in reg["sources"]
            )
            and r["candidate"]["nll"] <= r["base"]["nll"] + 0.03
        )
    if reg["id"] in (
        "all-layer-reasoning-01",
        "recurrent-deliberation-01",
        "projected-distillation-01",
        "privileged-reasoning-01",
        "late-recurrent-exit-01",
        "learned-exit-01",
        "learned-exit-confirm-01",
        "fullrank-reasoning-01",
    ):
        passed = (
            full["all"]["wins"] - full["all"]["losses"] >= reg.get("net_floor", 4)
            and full["all"]["candidate"]["seconds"] <= full["all"]["base"]["seconds"]
            and all(
                full[k]["wins"] > full[k]["losses"]
                and short[k]["wins"] >= short[k]["losses"]
                for k in reg["sources"]
            )
            and r["candidate"]["nll"] <= r["base"]["nll"] + 0.03
        )
    require(decision["passed"] == passed, "Transfer gate")


def verify_memory_rows(file, items, cap, gain=None, *, allow_unselected=False):
    from loopqwen.identity import fingerprint

    rows = read(file)["records"]
    require(len(rows) == len(items), "Incomplete learned memory cohort")
    for r, item in zip(rows, items, strict=True):
        ids = item["prompt_ids"]
        require(
            r["input_sha256"] == fingerprint(ids)
            and r["prompt_tokens"] == len(ids)
            and r["ledger_seed"] == item["seed"]
            and r["seed"] == 0
            and r["gold"] == item["gold"]
            and r["greedy"]
            and r["max_tokens"] == cap
            and 0 < len(r["token_ids"]) <= cap
            and r["elapsed_s"] >= r["ttft_s"] > 0,
            "Memory request or timing",
        )
        answer = first_turn_answer(r["token_ids"], r["text"])
        require(
            r["answer"] == answer and r["correct"] == (answer == item["gold"]),
            "Memory first-turn score",
        )
        if gain is None:
            require("attention_memory" not in r, "Untouched Base prior")
        elif allow_unselected and "attention_memory" not in r:
            require(not r["located"], "Missing attention span cannot locate the answer")
        else:
            section = r["attention_memory"]
            require(
                section["length"] == len(ids)
                and 0 <= section["start"] < section["stop"] <= len(ids) - 64
                and section["gain"] == gain + max(0.0, math.log(len(ids) / 2048))
                and r["located"]
                == (
                    section["start"] <= item["gold_start"]
                    and section["stop"] >= item["gold_stop"]
                ),
                "Learned passage and gain",
            )
    return rows


def verify_attention_memory(root, decision):
    from loopqwen.identity import fingerprint

    reg = read(root / "registration.json")
    data = Path("research/data/attention-memory-01.json")
    d = read(data)
    require(
        sha(data) == decision["data"]
        and sha(root / "registration.json") == decision["registration"]
        and decision["parent"]
        == reg["parent_sha256"]
        == sha(Path(reg["parent"]) / "adapters.safetensors"),
        "Memory experiment identity",
    )
    if decision["status"] == "interrupted":
        require(
            not decision["passed"] and decision["reason"], "Interrupted memory study"
        )
        return
    training = read(root / "training.json")
    require(
        len(training) == reg["updates"]
        and d["training_examples"] == reg["updates"] * reg["logical_batch"]
        and sum(h["seconds"] for h in training) == decision["update_seconds"]
        and all(math.isfinite(h["loss"]) and h["loss"] >= 0 for h in training)
        and sha(root / "attention-reader.safetensors") == decision["scorer_weights"],
        "Memory fitting and scorer weights",
    )
    import struct

    recovery = read(root / "reader-recovery.json")
    require(
        recovery["updates"] == reg["updates"]
        and recovery["weights"] == decision["scorer_weights"]
        and recovery["optimizer"] == sha(root / "reader-optimizer.safetensors"),
        "Recovered passage scorer and Adam identities",
    )

    def tensor_header(file):
        header = json.loads(file.read(struct.unpack("<Q", file.read(8))[0]))
        header.pop("__metadata__", None)
        return header

    with (root / "attention-reader.safetensors").open("rb") as file:
        trainable = tensor_header(file)
    expected = {
        f"{key}.{moment}": v["shape"]
        for key, v in trainable.items()
        for moment in ("m", "v")
    }
    with (root / "reader-optimizer.safetensors").open("rb") as file:
        slots = tensor_header(file)
        require(
            set(slots) == set(expected) | {"step", "learning_rate"}
            and all(slots[k]["shape"] == shape for k, shape in expected.items()),
            "Passage scorer moment shapes",
        )
        cursor = slots["step"]
        require(
            cursor["dtype"] == "U64" and cursor["shape"] == [],
            "Passage Adam step schema",
        )
        file.seek(cursor["data_offsets"][0], 1)
        require(
            struct.unpack("<Q", file.read(8))[0] == recovery["updates"],
            "Passage Adam step",
        )
    cal = {}
    for gain in reg["gains"]:
        rows = verify_memory_rows(
            root / f"calibration-{gain}.json.gz",
            d["calibration"],
            reg["generation_cap"],
            gain,
        )
        cal[str(gain)] = {
            "correct": sum(r["correct"] for r in rows),
            "located": sum(r["located"] for r in rows),
            "seconds": sum(r["elapsed_s"] for r in rows),
        }
    best = max(reg["gains"], key=lambda g: (cal[str(g)]["correct"], -g))
    require(
        decision["calibration"] == cal and decision["gain"] == best,
        "Memory strength selection",
    )
    eligible = cal[str(best)]["correct"] >= 8 and cal[str(best)]["located"] >= 10
    if not eligible:
        require(
            decision["status"] == "calibration_failed"
            and not decision["passed"]
            and not (root / "base.json.gz").exists(),
            "Failed calibration closes evaluation",
        )
        return
    meta = read(root / "model/attention-reader.json")
    require(
        fingerprint(meta) == decision["weights"]
        and meta["scorer_sha256"]
        == sha(root / "model/attention-reader.safetensors")
        == decision["scorer_weights"]
        and meta["gain"] == best
        and meta["minimum"] == reg["activation_length"]
        and meta["adapter_sha256"]
        == sha(root / "model/adapters.safetensors")
        == reg["parent_sha256"],
        "Deployed memory checkpoint",
    )
    base = verify_memory_rows(root / "base.json.gz", d["panel"], reg["generation_cap"])
    candidate = verify_memory_rows(
        root / "candidate.json.gz", d["panel"], reg["generation_cap"], best
    )
    counts = {
        a: {
            "correct": sum(r["correct"] for r in rows),
            "seconds": sum(r["elapsed_s"] for r in rows),
        }
        for a, rows in (("base", base), ("candidate", candidate))
    }
    require(
        counts == decision["scores"]
        and decision["located"] == sum(r["located"] for r in candidate),
        "Memory score aggregates",
    )
    parent = Path(reg["parent"]).parent
    known = read(parent / "candidate.json.gz")["records"]
    short = read(root / "short.json.gz")["records"]
    require(
        decision["short_parity"] == len(short) == len(known) == 32
        and all(
            a["token_ids"] == b["token_ids"]
            and a["input_sha256"] == fingerprint(b["prompt_ids"])
            and "attention_memory" not in a
            for a, b in zip(short, known, strict=True)
        ),
        "Short ability parity",
    )
    base_nll = read(parent / "decision.json")["base"]["retention"]
    verify_retention_pair(base_nll, decision["retention"])
    gate = (
        counts["candidate"]["correct"] >= 7
        and sum(r["correct"] for r in candidate[4:]) >= 3
        and all(
            any(
                r["correct"] and d["panel"][i]["style"] == style
                for i, r in enumerate(candidate)
                if i >= 4
            )
            for style in ("table", "filing")
        )
        and counts["candidate"]["correct"] - counts["base"]["correct"] >= 2
        and counts["candidate"]["seconds"] <= 1.75 * counts["base"]["seconds"]
        and decision["retention"]["nll"] <= base_nll["nll"] + 0.03
    )
    require(decision["passed"] == gate, "Memory full-length qualification")


def verify_full_memory(root, decision):
    reg = read(root / "registration.json")
    path = Path("research/data") / (reg["id"] + ".json")
    data = read(path)
    parent = Path(reg["qualification"])
    qualification = read(parent / "decision.json")
    require(
        qualification["passed"]
        and sha(parent / "checksums.json")
        == reg["qualification_sha256"]
        == decision["qualification"]
        and sha(path) == decision["data"]
        and sha(root / "registration.json") == decision["registration"],
        "Full-context prerequisite and identities",
    )
    if "audit" in reg:
        from scripts.research_mixed_audit import qualified

        annotations = Path(reg["audit"])
        require(
            sha(annotations) == reg["audit_sha256"]
            and qualified(parent.name, read(annotations)),
            "Combined answer-content qualification",
        )
    first, second = data["panel"]
    a, b = first["prompt_ids"], second["prompt_ids"]
    (i, j), (k, l) = data["key_spans"]
    if decision["status"] == "invalid_intervention":
        require(
            not decision["passed"]
            and decision["reason"]
            and len(a) == len(b) >= reg["minimum_length"]
            and 0 < i < j < k < l < len(a)
            and j - i != l - k
            and a[:i] + a[k:l] + a[j:k] + a[i:j] + a[l:] == b,
            "Rejected unequal-width key intervention",
        )
        native = read(root / "native-decision.json")
        require(
            native["status"] == "completed"
            and native["registration"] == decision["registration"]
            and native["qualification"] == decision["qualification"]
            and native["data"] == decision["data"]
            and native["scores"] == decision["observed"],
            "Preserved native observations",
        )
        ref = read(Path(reg["parity_reference"]) / "decision.json")
        for arm in ("candidate", "base"):
            runs = verify_memory_rows(
                root / (arm + ".json.gz"),
                data["panel"],
                reg["generation_cap"],
                ref["gain"] if arm == "candidate" else None,
                allow_unselected=True,
            )
            require(
                len(runs) == 2
                and all(v["prompt_tokens"] == len(a) for v in runs)
                and native["scores"][arm]
                == {
                    "correct": sum(v["correct"] for v in runs),
                    "seconds": sum(v["elapsed_s"] for v in runs),
                },
                "Actual full-history responses in rejected intervention",
            )
        return
    require(
        0 < i < j < k < l < len(a)
        and a[i:j] != a[k:l]
        and j - i == l - k
        and a[:i] + a[k:l] + a[j:k] + a[i:j] + a[l:] == b,
        "Only the two equal-length keys may move",
    )
    require(
        first["gold"] != second["gold"]
        and first["key"] == second["key"]
        and reg["minimum_length"] <= len(a) == len(b) <= reg["minimum_length"] + 48
        and len(set(data["swapped_keys"])) == 2
        and data["swapped_keys"][0] == first["key"],
        "Counterfactual labels and context length",
    )
    require(
        0.08 < first["gold_start"] / len(a) < 0.12
        and 0.73 < second["gold_start"] / len(b) < 0.77
        and all(
            a[r["gold_start"] : r["gold_stop"]] == b[r["gold_start"] : r["gold_stop"]]
            for r in (first, second)
        ),
        "Unchanged values at separated depths",
    )
    if decision["status"] == "interrupted":
        require(
            not decision["passed"] and decision["reason"],
            "Interrupted context comparison",
        )
        return
    probe = read(root / "disk-parity.json.gz")["records"][0]
    reference = Path(reg.get("parity_reference", parent))
    if "parity_reference" in reg:
        require(
            sha(reference / "checksums.json") == reg["parity_reference_sha256"],
            "Full-context parity reference identity",
        )
    ref = read(reference / "candidate.json.gz")["records"][0]
    parity = (
        probe["token_ids"] == ref["token_ids"]
        and probe.get("attention_memory") == ref["attention_memory"]
        and probe["input_sha256"] == ref["input_sha256"]
        and probe["scratch_bytes"] > 0
    )
    mode = reg.get("cache_mode", "disk")
    require(mode in ("resident", "disk"), "Known full-history cache mode")
    if mode == "resident":
        resident = read(root / "resident-parity.json.gz")["records"][0]
        parity = (
            parity
            and "scratch_bytes" not in resident
            and all(
                resident.get(k) == probe.get(k)
                for k in (
                    "input_sha256",
                    "token_ids",
                    "attention_memory",
                    "context_route",
                )
            )
        )
    if decision["status"] == "disk_parity_failed":
        require(
            not parity
            and not decision["passed"]
            and not (root / "base.json.gz").exists(),
            "Failed parity closes context job",
        )
        return
    require(parity, "Validated disk reader required")
    rows = {
        arm: verify_memory_rows(
            root / (arm + ".json.gz"),
            data["panel"],
            reg["generation_cap"],
            read(reference / "decision.json")["gain"] if arm == "candidate" else None,
            allow_unselected="parity_reference" in reg,
        )
        for arm in ("base", "candidate")
    }
    counts = {
        arm: {
            "correct": sum(r["correct"] for r in rs),
            "seconds": sum(r["elapsed_s"] for r in rs),
        }
        for arm, rs in rows.items()
    }
    require(
        all(
            (
                r.get("scratch_bytes", 0) > 0
                if mode == "disk"
                else "scratch_bytes" not in r
            )
            and r["peak_bytes"] > 0
            for rs in rows.values()
            for r in rs
        )
        and counts == decision["scores"],
        "Full-history costs and counts",
    )
    a, c = counts["base"], counts["candidate"]
    require(
        decision["passed"]
        == (
            c["correct"] == 2
            and c["correct"] > a["correct"]
            and c["seconds"] <= 1.75 * a["seconds"]
            and all(r["located"] for r in rows["candidate"])
        ),
        "Counterfactual full-context gate",
    )


def verify_natural_context(root, decision):
    from loopqwen import REVISION
    from loopqwen.evaluation import extractive
    from loopqwen.identity import fingerprint

    reg = read(root / "registration.json")
    panel = Path("research/data/natural-context-01.json")
    train = Path("research/data/natural-context-training.json")
    items = read(panel)
    schedule = read(train)
    require(
        sha(root / "registration.json") == decision["registration"]
        and sha(panel) == decision["data"]
        and sha(train) == decision["training"]
        and len(items) == len(reg["titles"]) == 24
        and len(set(schedule["ids"])) == reg["passages"]
        and set(schedule["ids"]).isdisjoint(r["id"] for r in items)
        and [r["title"] for r in items] == reg["titles"]
        and sha(Path(reg["parent"]).parent / "checksums.json") == reg["qualification"]
        and sha(Path(reg["dense_control"]) / "adapters.safetensors")
        == reg["dense_sha256"],
        "Natural source or controls",
    )
    if decision["status"] == "interrupted":
        require(
            not decision["passed"] and decision["reason"], "Interrupted natural study"
        )
        return
    model = root / "model"
    policy = read(model / "context-policy.json")
    st = read(root / "prose-training.json")
    reader = read(model / "reader-training.json")
    require(
        sha(model / "context-policy.json") == decision["weights"]
        and all(sha(model / k) == v for k, v in policy["files"].items())
        and sha(model / "adapters.safetensors")
        == sha(Path(reg["parent"]) / "adapters.safetensors")
        and st["parent"] == sha(model / "adapters.safetensors")
        and st["step"] == reg["updates"]
        and sum(h["targets"] for h in st["history"]) == schedule["targets"]
        and set(reader["questions"]) <= set(schedule["ids"])
        and len(set(reader["questions"]))
        == len(reader["loss"])
        >= 0.75 * reg["passages"]
        and reader["route_examples"] == 3 * reg["routing_examples"]
        and reader["route_updates"] == 48,
        "Natural training lineage",
    )
    scores = {}
    for arm in ("base", "candidate"):
        blob = read(root / (arm + ".json.gz"))
        require(
            blob["weights"] == (REVISION if arm == "base" else decision["weights"]),
            "Decoder identity",
        )
        rows = blob["records"]
        for v, r in zip(rows, items, strict=True):
            ids = v["token_ids"]
            answer = extractive.answer(ids, v["text"])
            em, f1 = extractive.score(answer, r["aliases"])
            sec = v.get("attention_memory")
            lo, hi = r["source_interval"]
            require(
                v["input_sha256"] == fingerprint(r["prompt_ids"])
                and v["id"] == r["id"]
                and v["prompt_tokens"] == len(r["prompt_ids"]) >= reg["minimum_length"]
                and v["greedy"]
                and v["seed"] == 0
                and v["max_tokens"] == reg["generation_cap"]
                and 0 < len(ids) <= v["max_tokens"]
                and v["elapsed_s"] >= v["ttft_s"] > 0
                and v["answer"] == answer
                and v["correct"] == em
                and v["f1"] == f1
                and v["located"]
                == bool(sec and lo < sec["stop"] and hi > sec["start"]),
                "Paired natural evidence, score, or timing",
            )
            if arm == "candidate":
                require(
                    v["context_route"]
                    in ("prose_recurrent", "register_recurrent", "ordinary"),
                    "Unrecognized natural route",
                )
        scores[arm] = {
            "exact": sum(v["correct"] for v in rows),
            "f1": sum(v["f1"] for v in rows) / len(rows),
            "seconds": sum(v["elapsed_s"] for v in rows),
            "located": sum(v["located"] for v in rows),
        }
    a, c = scores["base"], scores["candidate"]
    require(
        scores == decision["scores"]
        and decision["passed"]
        == (
            c["exact"] >= a["exact"] + 3
            and c["f1"] >= a["f1"]
            and c["seconds"] <= 1.25 * a["seconds"]
        ),
        "Natural comparison gate",
    )


def verify_prose_reading(root, decision):
    import string
    from collections import Counter

    from loopqwen import REVISION
    from loopqwen.identity import fingerprint

    reg = read(root / "registration.json")
    data = Path("research/data/prose-reading-01.json")
    items = read(data)
    parent = Path(reg["qualification"])
    require(
        sha(data) == decision["data"]
        and sha(root / "registration.json") == decision["registration"]
        and sha(parent / "checksums.json")
        == reg["qualification_sha256"]
        == decision["qualification"]
        and read(parent / "decision.json")["passed"]
        and [i["title"] for i in items] == reg["titles"]
        and len({i["id"] for i in items}) == 12,
        "Public reading prerequisites",
    )
    if decision["status"] == "interrupted":
        require(
            not decision["passed"] and decision["reason"],
            "Interrupted reading measurement",
        )
        return

    def words(text):
        s = "".join(c for c in text.lower() if c not in string.punctuation)
        return [
            w for w in re.split(r"\s+", re.sub(r"\b(a|an|the)\b", " ", s).strip()) if w
        ]

    counts = {}
    hits = 0
    for arm in ("base", "candidate"):
        blob = read(root / (arm + ".json.gz"))
        require(
            blob["weights"]
            == (REVISION if arm == "base" else reg["qualification_sha256"]),
            "Public reading weights",
        )
        rows = blob["records"]
        for r, item in zip(rows, items, strict=True):
            ids = r["token_ids"]
            require(
                r["input_sha256"] == fingerprint(item["prompt_ids"])
                and reg["minimum_length"]
                <= r["prompt_tokens"]
                == len(item["prompt_ids"])
                and r["id"] == item["id"]
                and r["greedy"]
                and r["seed"] == 0
                and r["max_tokens"] == reg["generation_cap"]
                and 0 < len(ids) <= r["max_tokens"]
                and r["elapsed_s"] >= r["ttft_s"] > 0,
                "Public reading prompt, policy and timing",
            )
            ans = None
            if (
                ids[-1] in (248044, 248046)
                and ids.count(248069) == 1
                and 248068 not in ids
            ):
                boundary = "<|im_end|>" if ids[-1] == 248046 else "<|endoftext|>"
                require(
                    not set(ids[:-1]).intersection((248044, 248046))
                    and r["text"].endswith(boundary)
                    and r["text"].count("</think>") == 1,
                    "Free-text turn boundary",
                )
                raw = r["text"].removesuffix(boundary).split("</think>")[1].strip()
                ans = (
                    re.sub(
                        r"^(?:final\s+)?answer\s*:\s*",
                        "",
                        raw,
                        count=1,
                        flags=re.IGNORECASE,
                    ).strip()
                    or None
                )
            target = [words(x) for x in item["aliases"]]
            predicted = words(ans or "")
            em = ans is not None and predicted in target
            f1 = 0
            for gold in target:
                overlap = sum((Counter(predicted) & Counter(gold)).values())
                if overlap:
                    f1 = max(f1, 2 * overlap / (len(predicted) + len(gold)))
            require(
                (ans, em, f1) == (r["answer"], r["correct"], r["f1"]),
                "Independent public EM and F1",
            )
            if arm == "candidate":
                sec = r["attention_memory"]
                lo, hi = item["source_interval"]
                hit = sec["start"] < hi and lo < sec["stop"]
                require(hit == r["selected_source"], "Selected public source paragraph")
                hits += hit
            else:
                require("attention_memory" not in r, "Public Base reader")
        counts[arm] = {
            "exact": sum(r["correct"] for r in rows),
            "f1": sum(r["f1"] for r in rows) / len(items),
            "seconds": sum(r["elapsed_s"] for r in rows),
            "completed": sum(r["answer"] is not None for r in rows),
            "placement": {
                p: sum(r["correct"] for r, i in zip(rows, items) if i["placement"] == p)
                for p in ("early", "middle", "late")
            },
        }
    a, c = counts["base"], counts["candidate"]
    require(
        counts == decision["scores"]
        and hits == decision["selected_source"]
        and decision["passed"]
        == (
            c["exact"] >= 8
            and c["exact"] - a["exact"] >= 2
            and c["f1"] >= a["f1"]
            and c["seconds"] <= 1.5 * a["seconds"]
        ),
        "Natural reading gate",
    )


def verify_shared_budget(root, decision):
    from collections import Counter

    from loopqwen import REVISION

    reg = read(root / "registration.json")
    data = Path("research/data/shared-budget-vote-01.json")
    items = read(data)
    require(
        sha(data) == decision["data"]
        and sha(root / "registration.json") == decision["registration"]
        and len(items) == 2 * reg["per_source"]
        and len({r["id"] for r in items}) == len(items),
        "Shared budget panel and registration",
    )
    if decision["status"] == "interrupted":
        require(
            not decision["passed"] and decision["reason"],
            "Interrupted reasoning measurement",
        )
        return
    arms = {}
    for arm, weight in (("base", REVISION), ("candidate", reg["parent_sha256"])):
        blob = read(root / (arm + ".json.gz"))
        require(blob["weights"] == weight, "Shared-budget model")
        rows = blob["records"]
        for i, (r, item) in enumerate(zip(rows, items, strict=True)):
            require(
                r["id"] == item["id"]
                and r["gold"] == item["gold"]
                and r["prompt_ids"] == item["prompt_ids"],
                "Shared-budget matched input",
            )
            left = reg["generation_cap"]
            counts = Counter()
            done = False
            choice = None
            traces = r["traces"]
            require(0 < len(traces) <= reg["max_attempts"], "Reasoning attempt count")
            for j, t in enumerate(traces):
                ids = t["token_ids"]
                require(
                    not done
                    and 0 < len(ids) <= left
                    and t["seconds"] >= t["ttft"] > 0
                    and t["seed"] == reg["seed"] + i * reg["max_attempts"] + j
                    and t["temperature"] == (reg["temperature"] if j else 0),
                    "Attempt identity and cost",
                )
                boundary = ids[-1] in (248044, 248046)
                require(
                    not set(ids[:-1]).intersection((248044, 248046))
                    and (boundary or len(ids) == left)
                    and t["stop"]
                    == (
                        "chat_end"
                        if ids[-1] == 248046
                        else "eos"
                        if boundary
                        else "cap"
                    )
                    and t["complete"]
                    == (boundary and 248068 not in ids and ids.count(248069) == 1),
                    "Every attempt spends its available budget or stops at its first boundary",
                )
                left -= len(ids)
                answer = first_turn_answer(ids, t["text"])
                require(answer == t["answer"], "Independent attempt answer")
                if answer is not None:
                    counts[answer] += 1
                choice = max(counts, key=counts.get) if counts else None
                remaining = reg["max_attempts"] - j - 1
                other = max((v for k, v in counts.items() if k != choice), default=0)
                done = (
                    left == 0
                    or remaining == 0
                    or (choice is not None and counts[choice] > other + remaining)
                )
            require(
                done
                and r["answer"] == choice
                and r["correct"] == (choice == item["gold"])
                and r["tokens"] == reg["generation_cap"] - left
                and r["seconds"] == sum(t["seconds"] for t in traces),
                "Vote final decision",
            )
        arms[arm] = rows
    for src, summ in decision["scores"].items():
        idx = [
            i for i, item in enumerate(items) if src == "all" or item["source"] == src
        ]
        wins = losses = 0
        for arm, rows in arms.items():
            rs = [rows[i] for i in idx]
            first = [r["traces"][0]["answer"] == r["gold"] for r in rs]
            require(
                summ[arm]
                == {
                    "correct": sum(r["correct"] for r in rs),
                    "valid": sum(r["answer"] is not None for r in rs),
                    "tokens": sum(r["tokens"] for r in rs),
                    "seconds": sum(r["seconds"] for r in rs),
                    "attempts": sum(len(r["traces"]) for r in rs),
                    "first_correct": sum(first),
                    "first_seconds": sum(r["traces"][0]["seconds"] for r in rs),
                    "fixed": sum(r["correct"] and not f for r, f in zip(rs, first)),
                    "lost": sum(not r["correct"] and f for r, f in zip(rs, first)),
                },
                "Vote performance summary",
            )
        for i in idx:
            a, c = arms["base"][i]["correct"], arms["candidate"][i]["correct"]
            wins += c and not a
            losses += a and not c
        n = wins + losses
        p = min(
            1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
        )
        require(
            (summ["wins"], summ["losses"], summ["p"]) == (wins, losses, p),
            "Vote paired outcomes",
        )
    scores = decision["scores"]
    pooled = scores["all"]
    require(
        decision["passed"]
        == (
            pooled["p"] < 0.05
            and pooled["candidate"]["seconds"] <= pooled["base"]["seconds"]
            and all(
                scores[k]["wins"] > scores[k]["losses"]
                and scores[k]["candidate"]["correct"]
                >= scores[k]["base"]["first_correct"]
                for k in reg["sources"]
            )
        ),
        "Shared budget gate",
    )


def verify_prefill(root, decision):
    from statistics import median

    from loopqwen.identity import fingerprint

    reg = read(root / "registration.json")
    source = Path("research/sessions/2026-11-25/context-128k-01/prompt.json.gz")
    ids = read(source)["prompt_ids"]
    require(
        sha(source) == decision["source_sha256"]
        and sha(root / "registration.json") == decision["registration"]
        and sha(Path(reg["candidate"]) / "adapters.safetensors")
        == decision["weights"]
        == reg["weights"],
        "Performance identity",
    )
    rows = read(root / "runs.json") if (root / "runs.json").exists() else []
    schedule = [
        (n, i, size)
        for n in reg["lengths"]
        for i, size in enumerate(
            [reg["baseline"], reg["treatment"], reg["treatment"], reg["baseline"]]
        )
    ]
    require(len(rows) <= len(schedule), "Extra performance run")
    for row, (n, i, size) in zip(rows, schedule):
        require(
            (row["prompt_tokens"], row["repeat"], row["block"]) == (n, i, size)
            and row["input_sha256"] == fingerprint(ids[:n])
            and row["max_tokens"] == reg["cap"]
            and row["greedy"]
            and row["elapsed_s"] >= row["ttft_s"] > 0
            and row["peak_bytes"] > 0,
            "Performance workload",
        )
        require(
            len(row["vocabulary_sha256"]) == len(row["token_ids"]) > 0
            and all(re.fullmatch("[0-9a-f]{64}", x) for x in row["vocabulary_sha256"])
            and all(
                math.isclose(x, math.sqrt(a / b), rel_tol=1e-12)
                for x, (a, b) in zip(
                    row["relative_rms"], row["squared_norms"], strict=True
                )
            ),
            "Logit norms or digests",
        )
    if decision["status"] == "interrupted":
        require(
            not decision["passed"] and decision["reason"], "Incomplete performance gate"
        )
        return
    require(len(rows) == len(schedule), "Incomplete crossover")
    for n in reg["lengths"]:
        selected = [x for x in rows if x["prompt_tokens"] == n]
        report = decision["lengths"][str(n)]
        for mode in ("baseline", "treatment"):
            times = [r for r in selected if r["block"] == reg[mode]]
            require(
                report[mode]
                == {
                    "ttft": median(r["ttft_s"] for r in times),
                    "peak_bytes": max(r["peak_bytes"] for r in times),
                },
                "Performance medians and peak",
            )
        same = all(x["token_ids"] == selected[0]["token_ids"] for x in selected)
        rms = max(max(x["relative_rms"]) for x in selected)
        ref, alt = report["baseline"], report["treatment"]
        require(
            (report["identical"], report["max_rms"], report["passed"])
            == (
                same,
                rms,
                same
                and rms <= 0.005
                and alt["ttft"] <= 0.90 * ref["ttft"]
                and alt["peak_bytes"] <= ref["peak_bytes"] + 0.30 * 2**30,
            ),
            "Performance length gate",
        )
    require(
        decision["passed"] == all(v["passed"] for v in decision["lengths"].values()),
        "Performance promotion",
    )


def verify_ledger_curriculum(root, decision):
    from loopqwen.identity import fingerprint

    reg = read(root / "registration.json")
    cohort = Path("research/data/ledger-curriculum-01.json")
    require(
        sha(root / "registration.json") == decision["registration"]
        and sha(cohort) == decision["data"],
        "Ledger registration and frozen requests",
    )
    items = read(cohort)["items"]
    require(
        len(items) == len({r["seed"] for r in items}) == 8
        and {(r["minimum_length"], r["fraction"], r["announce"]) for r in items}
        == {
            (n, f, a) for n in (2048, 16384) for f in (0.1, 0.75) for a in (False, True)
        },
        "Ledger evaluation design",
    )
    if "weights" in decision:
        state = read(root / "model/state.json")
        require(
            sha(root / "model/adapters.safetensors") == decision["weights"]
            and state["step"] == decision["updates"] == reg["updates"]
            and state["parent"] == reg["parent_sha256"]
            and decision["targets"] == sum(h["targets"] for h in state["history"]),
            "Ledger checkpoint lineage or exposure",
        )
    outputs = {}
    for arm in ("base", "candidate"):
        file = root / (arm + ".json.gz")
        if not file.exists():
            require(decision["status"] == "interrupted", "Missing ledger arm")
            continue
        rows = read(file)["records"]
        require(len(rows) <= len(items), "Too many ledger generations")
        for row, item in zip(rows, items):
            require(
                row["input_sha256"] == fingerprint(item["prompt_ids"])
                and row["prompt_tokens"] == len(item["prompt_ids"])
                and item["minimum_length"]
                <= row["prompt_tokens"]
                <= item["minimum_length"] + 32
                and row["ledger_seed"] == item["seed"]
                and row["seed"] == 0
                and row["gold"] == item["gold"]
                and row["greedy"]
                and row["max_tokens"] == reg["generation_cap"]
                and 0 < len(row["token_ids"]) <= row["max_tokens"]
                and row["elapsed_s"] >= row["ttft_s"] > 0,
                "Ledger request, generation policy or cost",
            )
            answer = first_turn_answer(row["token_ids"], row["text"])
            require(
                row["answer"] == answer and row["correct"] == (answer == item["gold"]),
                "Ledger lookup score",
            )
        outputs[arm] = rows
    if decision["status"] == "interrupted":
        require(
            not decision["passed"] and not decision["qualified"] and decision["reason"],
            "Interrupted ledger",
        )
        return
    require(
        all(len(outputs[arm]) == len(items) for arm in ("candidate", "base")),
        "Incomplete ledger pair",
    )
    audit = decision["arithmetic"]
    parent = Path(reg["parent"]).parent
    require(
        sha(root / "arithmetic/base.json.gz") == sha(parent / "base.json.gz")
        and audit["base"] == read(parent / "decision.json")["base"],
        "Frozen arithmetic Base",
    )
    candidate = read(root / "arithmetic/candidate.json.gz")
    arithmetic = candidate["records"]
    ref = read(parent / "base.json.gz")["records"]
    require(
        candidate["weights"] == decision["weights"]
        and len(arithmetic) == len(ref) == 32,
        "Arithmetic checkpoint or size",
    )
    for a, b in zip(arithmetic, ref, strict=True):
        require(
            (a["id"], a["gold"], a["prompt_ids"])
            == (b["id"], b["gold"], b["prompt_ids"])
            and a["answer"] == first_turn_answer(a["token_ids"], a["text"])
            and a["correct"] == (a["answer"] == a["gold"]),
            "Arithmetic retention response",
        )
    require(
        sum(v["correct"] for v in arithmetic) == audit["candidate"]["correct"]
        and sum(v["seconds"] for v in arithmetic) == audit["candidate"]["seconds"],
        "Arithmetic retention totals",
    )
    verify_retention_pair(audit["candidate"]["retention"], audit["base"]["retention"])
    eligible = []
    for announce, name in ((False, "query_last"), (True, "query_first")):
        ix = [i for i, r in enumerate(items) if r["announce"] == announce]
        pair = {
            a: {
                "correct": sum(rows[i]["correct"] for i in ix),
                "seconds": sum(rows[i]["elapsed_s"] for i in ix),
            }
            for a, rows in outputs.items()
        }
        require(pair == decision["scores"][name], "Ledger group scores")
        if (
            pair["candidate"]["correct"] == 4
            and pair["candidate"]["seconds"] <= 1.75 * pair["base"]["seconds"]
            and audit["candidate"]["retention"]["nll"]
            <= audit["base"]["retention"]["nll"] + 0.03
        ):
            eligible.append(name)
    require(
        decision["qualified"] == eligible
        and decision["passed"] == bool(eligible)
        and decision["arithmetic_retained"] == (audit["candidate"]["correct"] >= 13),
        "Ledger scaling decision",
    )


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
                row["answer"]
                == visible
                == first_turn_answer(row["token_ids"], row["text"])
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
    verify_retention_pair(c["retention"], b["retention"])
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
        for arm, previous_weight in prev.items():
            folder = root / f"round{j}" / arm
            score(
                read(folder / "eval.json.gz"),
                arms[arm]["weights"],
                arms[arm]["functional"],
            )
            meta = read(folder / "model/state.json")
            require(
                meta["parent"] == previous_weight
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

                def priority(row, arm=arm, slot=slot, j=j):
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
