"""Train localized document attention and test three complete real documents on Metal."""

import argparse
import gc
import random
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten
from mlx_lm.utils import load_tokenizer

from ..document_reader import DocumentReader, SpanRanker, install
from ..evaluation.generation_eval import generate
from ..evaluation.language7 import base
from ..evaluation.mechanism import packed, unpack
from ..export_runtime import load_export
from ..identity import fingerprint
from ..io import digest, read, save
from ..model_loader import local_snapshot
from .core import Budget
from .paired_progress import resume
from .register_allocation import packed_evidence
from .registered import registered
from .repository_tree_field import grade


def fit(spec, root, budget):
    from scripts.document_frequency_material import training_documents

    started = perf_counter()
    tok = load_tokenizer(local_snapshot())
    ranker = SpanRanker()
    reader = DocumentReader(tok, ranker)
    bank = []
    train = read(spec["training"])
    for source, question, start, stop in training_documents(train):
        budget.check()
        encoded = tok._tokenizer(
            source, add_special_tokens=False, return_offsets_mapping=True
        )
        target = [
            i
            for i, (a, b) in enumerate(encoded["offset_mapping"])
            if b > start and a < stop
        ]
        features, spans, _ = reader.features(
            encoded["input_ids"], tok.encode(question, add_special_tokens=False)
        )
        mask = [a <= target[0] and target[-1] < b for a, b in spans]
        if any(mask):
            mx.eval(features)
            bank.append((features, mx.array(mask)))
    if len(bank) < spec["min_train"]:
        raise ValueError("Too few complete training spans")
    width = max(len(f) for f, _ in bank)
    features = mx.stack([mx.pad(f, ((0, width - len(f)), (0, 0))) for f, _ in bank])
    valid = mx.array([[i < len(f) for i in range(width)] for f, _ in bank])
    target = mx.stack([mx.pad(m, (0, width - len(m))) for _, m in bank])
    rng = random.Random(spec["seed"])
    optimizer = optim.Adam(learning_rate=spec["lr"])

    def loss(net, x, declared, gold):
        logits = net(x)
        return (
            mx.logsumexp(mx.where(declared, logits, -1e9), axis=1)
            - mx.logsumexp(mx.where(gold, logits, -1e9), axis=1)
        ).mean()

    grad = nn.value_and_grad(ranker, loss)
    logs = []
    for step in range(spec["updates"]):
        budget.check()
        ix = mx.array(rng.sample(range(len(bank)), spec["batch"]))
        val, g = grad(ranker, features[ix], valid[ix], target[ix])
        optimizer.update(ranker, g)
        mx.eval(ranker.parameters(), optimizer.state, val)
        if step % 50 == 0 or step == spec["updates"] - 1:
            logs.append([step + 1, val.item()])
    dest = root / "span-ranker.safetensors"
    mx.save_safetensors(str(dest), dict(tree_flatten(ranker.parameters())))
    save(
        root / "fit.json",
        {
            "rows": len(bank),
            "training": digest(Path(spec["training"])),
            "weights": digest(dest),
            "coefficients": ranker.weight.tolist(),
            "loss": logs,
            "seconds": perf_counter() - started,
        },
    )
    print("trained", len(bank), flush=True)


def measure(spec, w, side, root, budget):
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    mx.reset_peak_memory()
    start = perf_counter()
    if side == "candidate":
        model, tok = load_export(Path(spec["checkpoint"]))
        install(model, tok, root, spec["checkpoint"])
    else:
        model, tok, _ = base()
    ids = w["prompt_ids"]
    text = tok.decode(ids)
    assert tok.encode(text, add_special_tokens=False) == ids
    loaded = perf_counter() - start
    peak = mx.get_peak_memory()
    first = []

    def record(n):
        if n == len(ids) and not first:
            first.append(perf_counter() - start)
        budget.check()

    res = generate(
        model,
        tok,
        text,
        spec["max_tokens"],
        stop_ids=[tok.convert_tokens_to_ids("<|im_end|>")],
        progress=record,
        record_kv_history=True,
    )
    del res["prompt"]
    res.update(
        id=w["id"],
        input_sha256=fingerprint(ids),
        grade=grade(tok, res["token_ids"], w["gold"], "__no_alternate__"),
        cold_seconds=perf_counter() - start,
        cold_first=first[0],
        load_seconds=loaded,
        peak_bytes=max(peak, res["peak_bytes"]),
    )
    return res


def outcome(spec, panel, data):
    groups = {}
    complete = True
    enclosed = 0
    for side, rows in data.items():
        n = correct = content = 0
        for w, r in zip(panel, rows, strict=True):
            n += 1
            correct += int(r["grade"]["exact"])
            content += int(r["grade"]["content"])
            length = len(w["prompt_ids"])
            consumed = length + len(r["token_ids"]) - 1
            sites = {(v["layer"], v["turn"]) for v in r["kv_history"]}
            expected = {(i, 0) for i in (3, 7, 11, 15, 19, 23)}
            if side == "candidate":
                expected.update({(11, 1), (15, 1)})
                section = r.get("attention_memory", {})
                enclosed += int(
                    section.get("start", length)
                    <= w["gold_start"]
                    < w["gold_stop"]
                    <= section.get("stop", 0)
                )
                pages = r.get("page_sites", [])
                complete &= (
                    r.get("context_route") == "register"
                    and len(pages) == len(expected)
                    and all(
                        packed_evidence(
                            v, consumed, capacity=length + spec["max_tokens"] - 1
                        )
                        and v["reader_biased_queries"]
                        == (len(r["token_ids"]) if v["layer"] >= 8 else 0)
                        for v in pages
                    )
                )
            complete &= sites == expected and all(
                v["offset"] == v["retained"] == consumed for v in r["kv_history"]
            )
        groups[side] = {
            "n": n,
            "exact": correct,
            "content": content,
            "wall_s": sum(r["cold_seconds"] for r in rows),
            "first_s": sum(r["cold_first"] for r in rows),
            "peak": max(r["peak_bytes"] for r in rows),
            "prefill_tps": sum(r["prompt_tokens"] for r in rows)
            / sum(r["ttft_s"] for r in rows),
            "decode_tps": sum(len(r["token_ids"]) - 1 for r in rows)
            / sum(r["elapsed_s"] - r["ttft_s"] for r in rows),
        }
    a, b = groups["candidate"], groups["control"]
    gates = {
        "exact": a["exact"] == len(panel) and a["exact"] > b["exact"],
        "located": enclosed == len(panel),
        "histories": complete,
        **{k: a[k] < b[k] for k in ("wall_s", "first_s", "peak")},
        **{k: a[k] > b[k] for k in ("prefill_tps", "decode_tps")},
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "groups": groups,
        "located": enclosed,
    }


def verify(root, decision):
    from scripts.document_frequency_material import panel as rebuild
    from scripts.document_frequency_material import training as rebuild_training

    spec = read(root / "registration.json")
    assert spec == registered(spec["id"])
    material = read(spec["material"])
    assert material == rebuild() and read(spec["training"]) == rebuild_training()
    panel = material["panel"]
    fit = read(root / "fit.json")
    assert fit["weights"] == digest(root / "span-ranker.safetensors") and fit[
        "training"
    ] == digest(Path(spec["training"]))
    assert (
        fit["rows"] >= spec["min_train"]
        and len(fit["coefficients"]) == 7
        and fit["loss"][-1][0] == spec["updates"]
    )
    from ..worker import acquire

    acquire()
    tok = load_tokenizer(local_snapshot())
    ranker = SpanRanker()
    ranker.load_weights(str(root / "span-ranker.safetensors"), strict=True)
    assert ranker.weight.tolist() == fit["coefficients"]
    reader = DocumentReader(tok, ranker)
    data = {}
    for side in ("candidate", "control"):
        data[side] = unpack(root / (side + ".json.gz"))
        for w, r in zip(panel, data[side], strict=True):
            assert (
                r["id"] == w["id"]
                and r["input_sha256"] == w["sha256"] == fingerprint(w["prompt_ids"])
                and r["prompt_tokens"] == len(w["prompt_ids"])
            )
            if side == "candidate":
                selected = reader.select(w["prompt_ids"])
                if selected is not None:
                    assert (
                        selected == r["attention_memory"]
                        and r["context_route"] == "register"
                    )
            assert r["grade"] == grade(
                tok, r["token_ids"], w["gold"], "__no_alternate__"
            )
            assert (
                0 < r["load_seconds"] < r["cold_first"] <= r["cold_seconds"]
                and 0 < r["ttft_s"] <= r["elapsed_s"] <= r["cold_seconds"]
            )
            assert len(r["token_ids"]) <= spec["max_tokens"] and r["peak_bytes"] > 0
    assert outcome(spec, panel, data) == decision


def run(name):
    spec = registered(name)
    release = Path(spec["requires"])
    assert (
        read(release / "decision.json")["passed"]
        and digest(release / "checksums.json") == spec["prerequisite_sha256"]
    )
    directory = Path(spec["checkpoint"])
    assert all(
        digest(directory / k) == v
        for k, v in read(release / "export-files.json").items()
    )
    from scripts.document_frequency_material import check_source

    check_source()
    train = read(spec["training"])
    assert all(digest(Path(k)) == v for k, v in train["sources"].items())
    panel = read(spec["material"])["panel"]
    root = Path("outputs/research") / name
    root.mkdir(parents=True, exist_ok=True)
    save(root / "registration.json", spec)
    budget = Budget(name, spec["limits"])
    try:
        if not (root / "fit.json").exists():
            fit(spec, root, budget)
        sides = ("candidate", "control")
        data = resume(root, spec, panel, sides)
        for i, w in enumerate(panel):
            for side in sides if i % 2 == 0 else reversed(sides):
                if len(data[side]) > i:
                    continue
                r = measure(spec, w, side, root, budget)
                data[side].append(r)
                packed(root / (side + ".json.gz"), data[side])
                print(side, w["id"], r["grade"], r.get("attention_memory"), flush=True)
        save(root / "decision.json", outcome(spec, panel, data))
    finally:
        budget.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    run(p.parse_args().name)
