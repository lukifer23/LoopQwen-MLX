"""Qualify a one-field tokenizer repair on a complete offline model inventory."""

import shutil
import tempfile
from collections import Counter
from pathlib import Path

from ..evaluation.mechanism import unpack
from ..identity import fingerprint
from ..io import digest, read, save


def build(source, parent, destination):
    reference = read(parent / "export-files.json")
    assert reference == read(source / "tool-export-files.json")
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        stage = Path(tmp) / "bundle"
        for name, hash_value in reference.items():
            src, dst = source / name, stage / name
            assert digest(src) == hash_value
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        config = stage / "tokenizer_config.json"
        settings = read(config)
        assert settings["tokenizer_class"] == "Qwen2Tokenizer"
        settings["tokenizer_class"] = "Qwen3_5Tokenizer"
        save(config, settings)
        with (stage / "README.md").open("a") as f:
            f.write(
                "\nRelease 05 names the Qwen3.5 tokenizer explicitly for text containing combining marks.\n"
            )
        inventory = {name: digest(stage / name) for name in reference}
        save(stage / "tool-export-files.json", inventory)
        if destination.exists():
            if read(destination / "tool-export-files.json") != inventory or any(
                digest(destination / name) != h for name, h in inventory.items()
            ):
                raise FileExistsError("Existing tokenizer release differs")
        else:
            stage.rename(destination)
        return inventory


def retained(run, earlier):
    from scripts.field_cli_replay import identical

    for group in ("register", "medium_register", "short"):
        actual = run[group] if group == "short" else [run[group]]
        original = earlier[group] if group == "short" else [earlier[group]]
        for a, b in zip(actual, original, strict=True):
            for key in (
                "input_sha256",
                "prompt_tokens",
                "token_ids",
                "context_route",
                "kv_history",
                "attention_memory",
                "page_sites",
                "prefill_policy",
            ):
                if a.get(key) != b.get(key):
                    return False
            policy_a, policy_b = (
                {
                    k: v
                    for k, v in r.get("weight_precision", {}).items()
                    if k not in ("setup_s", "setup_peak")
                }
                for r in (a, b)
            )
            if policy_a != policy_b:
                return False
    return all(
        run["book"][k] == earlier["book"][k]
        for k in (
            "id",
            "input_sha256",
            "tokens",
            "route",
            "nll",
            "predicted",
            "retained",
        )
    ) and identical(run["tool"], earlier["tool"])


def outcome(data, material, spec):
    old = unpack(Path(spec["parent"]) / "candidate.json.gz")
    rows = material["unicode"]["rows"]
    reference = [w["ids"] for w in rows]
    summary = {}
    for side, runs in data.items():
        counts = Counter()
        outputs = [r["unicode"]["label_ids"] for r in runs]
        assert outputs[0] == outputs[1] and len(outputs[0]) == len(reference)
        for w, ids, expected in zip(rows, outputs[0], reference, strict=True):
            counts[w["locale"]] += int(ids != expected)
        question = material["unicode"]["question"]
        langs = [r["unicode"]["question"] for r in runs]
        assert all(
            r["token_ids"] == langs[0]["token_ids"]
            and r["initial_ids"] == langs[0]["initial_ids"]
            for r in langs
        )
        summary[side] = {
            "mismatches": sum(counts.values()),
            "by_locale": dict(counts),
            "total_label_tokens": sum(map(len, outputs[0])),
            "question_matches": langs[0]["initial_ids"] == question["initial_ids"],
            "question_answer": langs[0]["text"],
            "question_correct": langs[0]["answer"] == question["gold"],
            "reference_tokens": sum(map(len, reference)),
            "labels": len(rows),
            "retained": [retained(r, old[i]) for i, r in enumerate(runs)],
            "stream_seconds": sum(r["stream_seconds"] for r in runs),
            "register_peak": max(r["register"]["peak_bytes"] for r in runs),
            "rss": max(r["high_water_rss_bytes"] for r in runs),
        }
        assert all(
            r["unicode"]["source"] == fingerprint([w["text"] for w in rows])
            for r in runs
        )
    a, b = summary["candidate"], summary["control"]
    passed = (
        a["mismatches"] == 0
        and b["mismatches"] > 0
        and a["question_matches"]
        and not b["question_matches"]
        and all(a["retained"] + b["retained"])
        and a["stream_seconds"] <= b["stream_seconds"] * 1.10
        and a["register_peak"] <= b["register_peak"] + 2**20
        and a["rss"] <= b["rss"] + 128 * 2**20
    )
    return {"passed": bool(passed), "streams": summary}
