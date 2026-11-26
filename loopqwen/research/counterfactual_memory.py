"""Test a frozen learned attention reader on a full-context key swap."""

import argparse
import gc
import json
import re
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

from ..deployment import load_trained
from ..evaluation.language7 import base
from ..evaluation.mechanism import packed, unpack
from ..io import digest, read, save
from ..model_loader import local_snapshot
from . import core, ledger_curriculum, memory_data

REG = Path("research/experiments/attention-memory-128k.json")
DATA = Path("research/data/attention-memory-128k.json")
LINE = re.compile(
    r"^The filing code under ([a-z]+ [a-z]+ [a-z]+) is ([0-9]{6})\.$", re.MULTILINE
)


def material(t, s):
    item = memory_data.make_item(t, s["seed"], s["minimum_length"], 0.1, "filing")
    source = t.decode(item["prompt_ids"])
    lines = list(LINE.finditer(source))
    early = next(line for line in lines if line[1] == item["key"])
    # Keep both document length and every value at exactly the same token position.
    for idx in sorted(range(len(lines)), key=lambda i: abs(i / len(lines) - 0.75)):
        late = lines[idx]
        if late.start() <= early.end() or late[1] in item["hard_keys"]:
            continue
        altered = (
            source[: early.start(1)]
            + late[1]
            + source[early.end(1) : late.start(1)]
            + early[1]
            + source[late.end(1) :]
        )
        ids = t.encode(altered, add_special_tokens=False)
        if len(ids) != len(item["prompt_ids"]):
            continue
        gold = t.encode(late[2], add_special_tokens=False)
        at = [
            i
            for i, x in enumerate(ids)
            if x == gold[0] and ids[i : i + len(gold)] == gold
        ]
        if (
            len(at) == 1
            and item["prompt_ids"][at[0] : at[0] + len(gold)] == gold
            and abs(idx / len(lines) - 0.75) < 0.01
        ):
            if t.decode(ids) != altered:
                raise ValueError("Counterfactual token roundtrip")
            other = dict(
                item,
                fraction=idx / len(lines),
                gold=late[2],
                prompt_ids=ids,
                gold_start=at[0],
                gold_stop=at[0] + len(gold),
            )
            spans = []
            for line in (early, late):
                before = t.encode(source[: line.start(1) - 1], add_special_tokens=False)
                key_ids = t.encode(" " + line[1], add_special_tokens=False)
                if (
                    item["prompt_ids"][len(before) : len(before) + len(key_ids)]
                    != key_ids
                ):
                    raise ValueError("Key token boundary")
                spans.append([len(before), len(before) + len(key_ids)])
            return {
                "swapped_keys": [early[1], late[1]],
                "key_spans": spans,
                "panel": [item, other],
            }
    raise ValueError("No length-matched intervention near registered depth")


def run(freeze, registration=REG, dataset=DATA):
    s = read(registration)
    mode = s.get("cache_mode", "disk")
    if mode not in ("disk", "resident"):
        raise ValueError("Unknown full-history cache mode")
    parent = Path(s["qualification"])
    if (
        not read(parent / "decision.json")["passed"]
        or digest(parent / "checksums.json") != s["qualification_sha256"]
    ):
        raise ValueError("Missing qualified reader")
    if "audit" in s:
        from scripts.research_mixed_audit import qualified

        annotations = Path(s["audit"])
        if digest(annotations) != s["audit_sha256"] or not qualified(
            parent.name, read(annotations)
        ):
            raise ValueError("Combined content audit prerequisite failed")
    parity_source = Path(s.get("parity_reference", parent))
    if (
        "parity_reference" in s
        and digest(parity_source / "checksums.json") != s["parity_reference_sha256"]
    ):
        raise ValueError("Different prior parity reference")
    model = parent / "model"
    t = load_tokenizer(local_snapshot())
    data = material(t, s)
    if freeze and not dataset.exists():
        dataset.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    if read(dataset) != data:
        raise ValueError("Counterfactual material differs")
    if freeze:
        return
    index = read("research/registry.json")
    if (
        digest(dataset) != index["datasets"][str(dataset)]["sha256"]
        or digest(registration) != index["experiments"][s["id"]]["registration"]
    ):
        raise ValueError("Unregistered context inputs")
    root = core.OUT / s["id"]
    if root.exists():
        raise FileExistsError(root)
    core.assets()
    b = core.Budget(s["id"], s["limits"])
    scratch = core.OUT / "scratch"
    scratch.mkdir(exist_ok=True)
    rows = {"base": [], "candidate": []}
    result = {
        "registration": digest(registration),
        "data": digest(dataset),
        "qualification": s["qualification_sha256"],
        "passed": False,
    }
    try:
        m, t, _ = load_trained(model)
        existing = read(Path("research/data/attention-memory-01.json"))["panel"][0]
        expected = unpack(parity_source / "candidate.json.gz")["records"][0]
        probe = ledger_curriculum.measure(
            m, t, existing, s["generation_cap"], b, offload_dir=scratch
        )
        packed(root / "disk-parity.json.gz", {"records": [probe]})
        parity = True
        if mode == "resident":
            resident = ledger_curriculum.measure(m, t, existing, s["generation_cap"], b)
            packed(root / "resident-parity.json.gz", {"records": [resident]})
            parity = all(
                resident.get(k) == probe.get(k)
                for k in (
                    "input_sha256",
                    "token_ids",
                    "attention_memory",
                    "context_route",
                )
            )
        del m, t
        gc.collect()
        mx.clear_cache()
        if (
            not parity
            or probe["token_ids"] != expected["token_ids"]
            or probe.get("attention_memory") != expected["attention_memory"]
        ):
            result.update(status="disk_parity_failed")
        else:
            for i, order in enumerate((("candidate", "base"), ("base", "candidate"))):
                for arm in order:
                    m, t, _ = base() if arm == "base" else load_trained(model)
                    row = data["panel"][i]
                    v = ledger_curriculum.measure(
                        m,
                        t,
                        row,
                        s["generation_cap"],
                        b,
                        offload_dir=scratch if mode == "disk" else None,
                    )
                    if arm == "candidate":
                        sec = v.get("attention_memory")
                        v["located"] = (
                            sec is not None
                            and sec["start"] <= row["gold_start"]
                            and sec["stop"] >= row["gold_stop"]
                        )
                    rows[arm].append(v)
                    packed(root / (arm + ".json.gz"), {"records": rows[arm]})
                    print(i, arm, v["correct"], v["elapsed_s"], flush=True)
                    del m, t
                    gc.collect()
                    mx.clear_cache()
            scores = {k: ledger_curriculum.scores(rs) for k, rs in rows.items()}
            a, c = scores["base"], scores["candidate"]
            result.update(
                status="completed",
                scores=scores,
                passed=c["correct"] == 2
                and c["correct"] > a["correct"]
                and all(r["located"] for r in rows["candidate"])
                and c["seconds"] <= 1.75 * a["seconds"],
            )
    except (MemoryError, TimeoutError, InterruptedError) as exc:
        result.update(status="interrupted", reason=str(exc))
    finally:
        b.close()
    save(root / "decision.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--study", default=REG.stem)
    args = parser.parse_args()
    run(
        args.freeze,
        REG.with_name(args.study + ".json"),
        DATA.with_name(args.study + ".json"),
    )
