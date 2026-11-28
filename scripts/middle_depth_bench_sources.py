"""Freeze five canonical choice tasks and verify native scores using the upstream harness."""

import argparse
import hashlib
import importlib.metadata
import json
from math import fsum, isfinite
from pathlib import Path
from urllib.request import urlopen

from loopqwen.io import digest, read, save

DATA = Path("data/standard-five")
OUT = Path("docs/measurements/middle-depth/standard-five")
PLAN = Path("research/diagnostics/middle-depth/standard-five.md")
TASKS = {
    "arc_challenge": (
        "allenai/ai2_arc",
        "210d026faf9955653af8916fad021475a3f00453",
        "ARC-Challenge/test",
        "arc/arc_challenge",
    ),
    "hellaswag": (
        "Rowan/hellaswag",
        "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "data/validation",
        "hellaswag/hellaswag",
    ),
    "winogrande": (
        "allenai/winogrande",
        "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
        "winogrande_xl/validation",
        "winogrande/default",
    ),
    "openbookqa": (
        "allenai/openbookqa",
        "388097ea7776314e93a529163e0fea805b8a6454",
        "main/test",
        "openbookqa/openbookqa",
    ),
    "sciq": (
        "allenai/sciq",
        "2c94ad3e1aafab77146f384e23536f97a4849815",
        "data/test",
        "sciq/sciq",
    ),
}


def task_rows(name):
    import lm_eval
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.tasks._yaml_loader import load_yaml

    if importlib.metadata.version("lm_eval") != "0.4.13":
        raise ValueError("Incorrect harness version")
    repo, revision, filename, template = TASKS[name]
    src = DATA / (name + ".parquet")
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}-00000-of-00001.parquet"
    if not src.exists():
        with urlopen(url, timeout=60) as stream:
            blob = stream.read()
        if not blob.startswith(b"PAR1") or not blob.endswith(b"PAR1"):
            raise ValueError("Invalid parquet download")
        src.parent.mkdir(exist_ok=True, parents=True)
        src.with_suffix(".tmp").write_bytes(blob)
        src.with_suffix(".tmp").replace(src)
    yaml = Path(lm_eval.__file__).parent / "tasks" / (template + ".yaml")
    cfg = load_yaml(yaml, resolve_func=True)
    split = filename.split("/")[-1]
    # Only the pinned held split is required for zero-shot requests.
    cfg.update(
        dataset_path="parquet",
        dataset_name=None,
        dataset_kwargs={"data_files": {split: str(src)}},
        training_split=None,
        validation_split=split if split == "validation" else None,
        test_split=split if split == "test" else None,
        num_fewshot=0,
    )
    task = ConfigurableTask(config=cfg)
    docs = task.eval_docs
    order = sorted(
        range(len(docs)),
        key=lambda i: hashlib.sha256(
            f"loopqwen-standard-4619/{name}/{i}".encode()
        ).digest(),
    )[:50]
    return (
        task,
        docs,
        order,
        dict(
            source=url,
            sha=digest(src),
            harness_yaml=digest(yaml),
            split=split,
            size=len(docs),
            indices=order,
        ),
    )


def run(arm=None):
    chunks = []
    manifest = {}
    scores = []
    key_audit = {}
    for name in TASKS:
        task, docs, order, meta = task_rows(name)
        manifest[name] = meta
        outcomes = read(OUT / arm / (name + ".json")) if arm else {}
        for i in order:
            doc = docs[i]
            req = task.construct_requests(
                doc, task.fewshot_context(doc, 0), metadata=(name, i, 1)
            )
            args = [list(x.args) for x in req]
            key = f"{name}/{i}"
            sig = hashlib.sha256(
                json.dumps(args, ensure_ascii=False).encode()
            ).hexdigest()
            chunks.append(dict(key=key, requests=args, sha=sig))
            choices = task.doc_to_choice(doc)
            gold = int(
                task.doc_to_text(doc)
                if task.multiple_input
                else task.doc_to_target(doc)
            )
            if name in ("arc_challenge", "openbookqa"):
                original = doc["choices"]["label"].index(doc["answerKey"].strip())
            elif name == "sciq":
                original = choices.index(doc["correct_answer"])
            elif name == "hellaswag":
                original = int(doc["label"])
            else:
                original = int(doc["answer"]) - 1
            if gold != original:
                raise ValueError(f"Upstream label disagrees with harness on {key}")
            key_audit[key] = gold
            if arm:
                row = outcomes[str(i)]
                if row["request_sha"] != sig:
                    raise ValueError("Canonical prompt changed")

                if len(row["choices"]) != len(req) or any(
                    v["target_tokens"] != len(v["token_logp"])
                    or not v["token_logp"]
                    or not all(isfinite(x) and x <= 0 for x in v["token_logp"])
                    for v in row["choices"]
                ):
                    raise ValueError("Invalid or missing likelihoods")
                vals = [fsum(v["token_logp"]) for v in row["choices"]]
                grades = task.process_results(doc, [(v, False) for v in vals])
                raw = max(range(len(vals)), key=lambda j: vals[j])
                norm = max(range(len(vals)), key=lambda j: vals[j] / len(choices[j]))
                if int(grades["acc"]) != (raw == gold) or (
                    "acc_norm" in grades and int(grades["acc_norm"]) != (norm == gold)
                ):
                    raise ValueError("Independent scorer and upstream disagree")
                scores.append(
                    dict(
                        key=key,
                        gold=gold,
                        raw=raw,
                        normalized=norm if "acc_norm" in grades else None,
                        correct=bool(grades.get("acc_norm", grades["acc"])),
                        raw_correct=bool(grades["acc"]),
                        seconds=row["seconds"],
                        peak=row["peak_mlx"],
                        input_positions=sum(
                            v["context_tokens"] + v["target_tokens"] - 1
                            for v in row["choices"]
                        ),
                        scored_tokens=sum(v["target_tokens"] for v in row["choices"]),
                        token_sha=row["token_sha"],
                        receipt=digest(OUT / arm / (name + ".json")),
                    )
                )
    key_sha = hashlib.sha256(json.dumps(key_audit, sort_keys=True).encode()).hexdigest()
    key_path = OUT / "label-verification.json"
    label_receipt = {
        "rows": len(key_audit),
        "label_sha": key_sha,
        "method": "Pinned dataset labels checked independently against all harness target indices.",
    }
    if key_path.exists() and read(key_path) != label_receipt:
        raise ValueError("Label verification changed")
    save(key_path, label_receipt)
    frozen = OUT / "enrollment.json"
    if arm:
        if read(frozen)["tasks"] != manifest or len(scores) != 250:
            raise ValueError("Enrollment changed")
        save(OUT / (arm + "-grades.json"), scores)
    else:
        requests = DATA / "requests.json"
        previous = read(frozen) if frozen.is_file() else None
        if previous and (
            previous["tasks"] != manifest
            or (requests.exists() and read(requests) != chunks)
        ):
            raise ValueError("Frozen standard benchmark changed")
        save(requests, chunks)
        save(
            frozen,
            dict(
                plan=digest(PLAN),
                tasks=manifest,
                requests=digest(requests),
                harness="0.4.13",
            ),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action", choices=("freeze", "base", "loop", "selected-loop", "loop-k1")
    )
    a = p.parse_args()
    run(None if a.action == "freeze" else a.action)
