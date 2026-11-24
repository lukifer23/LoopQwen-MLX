"""Bounded zero-shot readout and ordinary one-token generation on identical verbal tasks."""

from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import mlx.core as mx

from .. import REVISION
from ..data import retention3
from ..data.pointer5 import rows
from ..io import digest, read, save
from ..looped_model import LoopedModel
from ..model_loader import load_base
from ..recurrence import Span
from ..training.losses import validation_nll
from ..training.pointer5 import OUT, inputs, outputs
from .pointer5 import installed, milestone
from .pointer5_diagnostics import load


def text(row, depth):
    links = ". ".join(f"Room {i} leads to room {j}" for i, j in enumerate(row["map"]))
    hops = "" if depth is None else f", follow {depth} links"
    return f"{links}. Starting in room {row['start']}{hops}. Destination"


def transfer(seed):
    if not installed(seed, "recurrent"):
        raise ValueError("Direct installation did not qualify")
    outcome = milestone(seed, "outcome", 560)
    baselines = read(Path("data/phase5_pointer/manifest.json"))["prior"]
    if any(
        m["accuracy"] < max(p["conditional"], p["prior"], p["majority"]) + 0.15
        for m, p in zip(outcome["metrics"], baselines, strict=True)
    ):
        print("Persistence threshold missed; transfer deferred.")
        return
    retention3.build()
    prose = defaultdict(list)
    lang = Path("data/v3_retention/validation.jsonl")
    for line in lang.read_bytes().splitlines():
        row = json.loads(line)
        if len(prose[row["category"]]) < 4:
            prose[row["category"]].append(row)
    docs = [r for group in prose.values() for r in group]
    cohort = rows("development")[:96]
    for arm in ("base", "middle", "recurrent", "outcome"):
        if arm == "base":
            raw, tok, _ = load_base()
            m = LoopedModel(raw, Span(passes=1))
            weights = REVISION
        else:
            m, tok, weights = load(seed, arm, 560 if arm == "outcome" else 480)
        m.eval()
        path = OUT / str(seed) / f"transfer-{arm}.json"
        ident = dict(
            weights=weights,
            source=digest(Path(__file__)),
            registry=digest(Path("experiments/phase5_transfer.json")),
            data=digest(Path("data/phase5_pointer/development.jsonl")),
            retention=digest(lang),
        )
        if path.exists():
            if read(path)["identity"] != ident:
                raise ValueError("Incompatible published transfer result")
            continue
        records = []
        start = perf_counter()
        for n in (1, 2, 3):
            recurrent = arm in ("recurrent", "outcome")
            for i in range(0, len(cohort), 8):
                group = cohort[i : i + 8]
                gold = [r["states"][n - 1] for r in group]

                def encode(texts):
                    ids = [tok.encode(s, add_special_tokens=False) for s in texts]
                    return inputs([{"ids": v} for v in ids])

                cat = [None] * len(group)
                if arm != "base":
                    x, last = encode(
                        [text(r, None if recurrent else n) + ":" for r in group]
                    )
                    cat = mx.argmax(outputs(m, x, last, n, arm)[-1], axis=-1).tolist()
                prompts = [text(r, n) + " (one digit): " for r in group]
                x, last = encode(prompts)
                digits = [tok.encode(str(g), add_special_tokens=False) for g in gold]
                for prompt, g, ids in zip(prompts, gold, digits, strict=True):
                    if (
                        len(ids) != 1
                        or tok.encode(prompt + str(g), add_special_tokens=False)
                        != tok.encode(prompt, add_special_tokens=False) + ids
                    ):
                        raise ValueError("Expected one exact next-token digit boundary")
                for depth in sorted({1, n} if recurrent else {1}):
                    m.span = replace(m.span, passes=depth)
                    z = m.project(m.hidden(x)[mx.arange(len(group)), last]).astype(
                        mx.float32
                    )
                    pred = mx.argmax(z, axis=-1).tolist()
                    scores = mx.take_along_axis(z, mx.array(digits), axis=1).squeeze(
                        1
                    ) - mx.logsumexp(z, axis=1)
                    for j, (r, q, logp) in enumerate(
                        zip(group, pred, scores.tolist(), strict=True)
                    ):
                        records.append(
                            dict(
                                id=r["example_id"],
                                map=r["map"],
                                start=r["start"],
                                requested=n,
                                executed=depth,
                                gold=gold[j],
                                categorical=cat[j],
                                greedy_token_id=q,
                                greedy_text=tok.decode([q]),
                                gold_logp=logp,
                                correct=q == digits[j][0],
                                reference_token_id=digits[j][0],
                            )
                        )
        retention = {}
        for depth in (1, 3) if arm in ("recurrent", "outcome") else (1,):
            m.span = replace(m.span, passes=depth)
            retention[str(depth)] = validation_nll(m, docs, batch_size=1)
        save(
            path,
            dict(
                identity=ident,
                records=records,
                retention=retention,
                document_ids=[r["id"] for r in docs],
                seconds=perf_counter() - start,
            ),
        )
        del m
        if arm == "base":
            del raw
        mx.clear_cache()
