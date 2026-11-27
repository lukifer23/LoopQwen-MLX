"""Observe both exits of the same unchanged fitted recurrent forward on actual responses."""

import argparse
import math
from collections import Counter
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np

from loopqwen.cross_view import CrossViewAssistant
from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.reasoned_evidence_fit import LABELS

PROTOCOL = Path("research/diagnostics/proof-curriculum-readout-protocol.json")
DEST = Path("docs/measurements/proof-curriculum-visit-readout.json")


def summarize(rows):
    out = {}
    for family in ("all", "inline", "file", "wheel"):
        selected = [r for r in rows if family == "all" or r["family"] == family]
        valid = [r for r in selected if "exits" in r]
        delta = [r["delta_nats"] for r in valid]
        out[family] = {
            "enrolled": len(selected),
            "eligible": len(valid),
            "excluded": dict(
                Counter(r["excluded"] for r in selected if "excluded" in r)
            ),
            "active_matches_emitted": sum(
                r["exits"][1]["argmax"] == r["actual"] for r in valid
            ),
            "median_nats": float(np.median(delta)) if delta else None,
            "mean_nats": float(np.mean(delta)) if delta else None,
            "gold_odds_improved": sum(v > 0 for v in delta),
            "margin_met": sum(v >= 0.25 for v in delta),
            "gold_highest_among_three": [
                sum(
                    max(range(3), key=r["exits"][i]["logits"].__getitem__)
                    == LABELS.index(r["gold"])
                    for r in valid
                )
                for i in (0, 1)
            ],
        }
    return out


def run(root, verify):
    spec = read(root / "registration.json")
    panel = read(spec["material"])
    original = unpack(root / "candidate.json.gz")
    if verify:
        doc = read(DEST)
        assert (
            doc["protocol"] == digest(PROTOCOL)
            and doc["compiled"] == digest(root / "cross-view.safetensors")
            and doc["source"] == digest(root / "candidate.json.gz")
        )
        expected = [
            (w["id"], w["family"], w["gold"])
            for w in panel["rows"]
            if w["family"] != "retention"
        ]
        assert expected == [(r["id"], r["family"], r["gold"]) for r in doc["rows"]]
        assert doc["groups"] == summarize(doc["rows"])
        idx = {w["id"]: i for i, w in enumerate(panel["rows"])}
        for r in doc["rows"]:
            if "exits" not in r:
                continue
            t = original[idx[r["id"]]]["requests"][0]["trajectory"]["turns"][-1]
            offset = r["answer_at"]
            assert (
                r["prefix_sha256"]
                == fingerprint(t["prompt_ids"] + t["token_ids"][:offset])
                and r["actual"] == t["token_ids"][offset]
            )
            gd = LABELS.index(r["gold"])
            odds = []
            for end in r["exits"]:
                vals = end["logits"]
                odds.append(
                    float(
                        vals[gd]
                        - np.logaddexp(*[v for i, v in enumerate(vals) if i != gd])
                    )
                )
            assert math.isclose(odds[1] - odds[0], r["delta_nats"], abs_tol=1e-6)
        print(doc["groups"])
        return
    if DEST.exists():
        raise FileExistsError("Readout is sealed; use --verify")
    limits = read(PROTOCOL)["limits"]
    start = perf_counter()
    mx.reset_peak_memory()
    actor = CrossViewAssistant(spec["checkpoint"], root)
    model, tok = actor._tool_model, actor.tokenizer
    cls = [tok.encode(w, add_special_tokens=False)[0] for w in LABELS]
    end = tok.convert_tokens_to_ids("</think>")
    atoms = []
    for w, entry in zip(panel["rows"], original, strict=True):
        assert w["id"] == entry["id"]
        if w["family"] == "retention":
            continue
        r = {k: w[k] for k in ("id", "family", "gold")}
        atoms.append(r)
        tr = entry["requests"][0]["trajectory"]
        t = tr["turns"][-1]
        ids = t["token_ids"]
        if tr["status"] != "answered" or end not in ids:
            r["excluded"] = (
                "incomplete"
                if tr["status"] != "answered"
                else "no closing reasoning delimiter"
            )
            continue
        offset = next(
            (
                i
                for i in range(ids.index(end) + 1, len(ids))
                if ids[i] not in tok.eos_token_ids
                and ids[i] != tok.convert_tokens_to_ids("<|im_end|>")
                and tok.decode([ids[i]]).strip()
            ),
            None,
        )
        if offset is None:
            r["excluded"] = "no nonblank visible answer token"
            continue
        prefix = t["prompt_ids"] + ids[:offset]
        exits = []

        def observe(event, visit, hidden, exits=exits):
            if event == "exit":
                assert visit == len(exits)
                logits = model.project(model.text.model.norm(hidden[:, -1, :]))[
                    0
                ].astype(mx.float32)
                exits.append(
                    {
                        "argmax": int(mx.argmax(logits).item()),
                        "logits": logits[mx.array(cls)].tolist(),
                    }
                )

        model.hidden(mx.array([prefix]), observer=observe)
        assert len(exits) == 2
        odds = []
        idx = LABELS.index(w["gold"])
        for h in exits:
            vals = h["logits"]
            odds.append(
                float(
                    vals[idx]
                    - np.logaddexp(*[v for i, v in enumerate(vals) if i != idx])
                )
            )
        r.update(
            prefix_sha256=fingerprint(prefix),
            positions=len(prefix),
            answer_at=offset,
            actual=ids[offset],
            exits=exits,
            delta_nats=odds[1] - odds[0],
        )
        assert (
            perf_counter() - start < limits["wall_seconds"]
            and mx.get_peak_memory() < limits["peak_gib"] * 2**30
        )
    save(
        DEST,
        {
            "protocol": digest(PROTOCOL),
            "compiled": digest(root / "cross-view.safetensors"),
            "source": digest(root / "candidate.json.gz"),
            "classes": dict(zip(LABELS, cls, strict=True)),
            "seconds": perf_counter() - start,
            "peak": mx.get_peak_memory(),
            "groups": summarize(atoms),
            "rows": atoms,
        },
    )
    run(root, True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify", action="store_true")
    p.add_argument(
        "--study-dir", type=Path, default=Path("research/results/proof-curriculum-01")
    )
    a = p.parse_args()
    run(a.study_dir, a.verify)
