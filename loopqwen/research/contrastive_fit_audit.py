"""Recount every optimizer label and replay generated training actions from the archived tokens."""

from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from safetensors import safe_open

from scripts.contrastive_world_material import REASON, TABLES

from ..evaluation.mechanism import unpack
from ..identity import fingerprint
from ..io import digest, read
from .conditional_fit import batches
from .contrastive_features import OLD
from .contrastive_fit import grade, index, selection, summarize
from .cross_view_study import tokenizer
from .native_function_data import token_pair
from .registered import registered
from .table_policy_learning import visible
from .table_worlds import prepare_tables


def verify(root, d):
    s = read(root / "registration.json")
    assert s == registered(s["id"])
    if d.get("status") == "interrupted":
        assert not d["passed"] and d["reason"]
        if "durable_updates" in d:
            assert read(root / "partial-state.json")["step"] == d["durable_updates"]
        return
    fit = read(root / "fit.json")
    idx, archives = index(s)
    assert fit["feature_archives"] == archives and fit["backbone"] == read(
        OLD / "backbone.json"
    )
    assert fit["registration"] == digest(root / "registration.json") and fit[
        "weights"
    ] == digest(root / "readout.safetensors")
    log = unpack(root / "training-log.json.gz")
    assert fit["updates"] == len(log)
    w = 1024
    r = s["rank"]
    g = s["gate_rank"]
    shapes = {
        "bias": (w,),
        "down": (4, r, w),
        "up.weight": (w, 4 * r),
        "context.down.weight": (g, w),
        "context.up.weight": (w, g),
    }
    assert fit["trainable"] == sum(int(np.prod(v)) for v in shapes.values())
    with (
        safe_open(root / "readout.safetensors", framework="np") as params,
        safe_open(root / "optimizer.safetensors", framework="np") as opt,
    ):
        assert set(params.keys()) == set(shapes) and set(opt.keys()) == {
            "learning_rate",
            "step",
        } | {k + "." + v for k in shapes for v in ("m", "v")}
        assert opt.get_tensor("step").item() == len(log) and np.isclose(
            opt.get_tensor("learning_rate"), s["lr"]
        )
        for key, shape in shapes.items():
            for arr in (
                params.get_tensor(key),
                opt.get_tensor(key + ".m"),
                opt.get_tensor(key + ".v"),
            ):
                assert (
                    arr.shape == shape
                    and arr.dtype == np.float32
                    and np.isfinite(arr).all()
                    and np.any(arr != 0)
                )
            assert np.all(opt.get_tensor(key + ".v") >= 0)
    for i, (line, micros) in enumerate(zip(log, batches(idx, s), strict=True), 1):
        members = [item for group in micros for item in group]
        assert (
            line["step"] == i
            and line["turns"] == len(members)
            and line["targets"] == sum(idx[j]["positions"] for j, _ in members)
        )
        assert line["preferences"] == sum(n is not None for _, n in members)
        assert np.isclose(
            line["objective_sum"],
            line["nll_sum"]
            + s["teacher_kl"] * line["teacher_kl_sum"]
            + s["preference_weight"] * line["preference_loss_sum"],
            atol=1e-3,
        )
        assert line["teacher_kl_sum"] >= -1e-3 and line["preference_loss_sum"] >= 0
    prob = read(root / "training-probes.json")
    groups = defaultdict(list)
    for rec in idx:
        groups[rec["group"]].append(rec)
    expected = [
        item
        for name in sorted(groups)
        for item in sorted(
            groups[name], key=lambda r: fingerprint([s["fit_seed"], "probe", r["id"]])
        )[:8]
    ]
    for p, item in zip(prob, expected, strict=True):
        assert (p["id"], p["group"], p["positions"]) == (
            item["id"],
            item["group"],
            item["positions"],
        )
        assert (
            type(p["collapsed_equal"]) is bool and min(p["trained"], p["ordinary"]) >= 0
        )
    rows = selection(unpack(Path(s["training"])), s)
    tok = tokenizer(s)
    stops = set(tok.eos_token_ids) | {tok.convert_tokens_to_ids("<|im_end|>")}
    free = unpack(root / "training-free-generation.json.gz")
    worlds = {r["id"]: r for r in unpack(TABLES / "training.json.gz")}
    qa = {r["id"]: r for r in unpack(REASON / "training.json.gz")}
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        prepare_tables(
            folder, [worlds[r["pair"]] for r in rows if r["group"] == "operator"]
        )
        for r, record in zip(rows, free, strict=True):
            assert (
                all(r[k] == record[k] for k in ("id", "pair", "group"))
                and record["opening"] == token_pair(tok, r)[0]
            )
            assert (
                0
                < len(record["tokens"])
                <= s["free_sql_cap" if r["group"] == "operator" else "free_answer_cap"]
            )
            assert not any(t in stops for t in record["tokens"][:-1]) and visible(
                tok, record, stops
            ) == (record["visible"], record["closed"])
            assert grade(record, r, worlds, qa, folder) == record["score"]
    assert summarize(s, log, prob, free) == d
