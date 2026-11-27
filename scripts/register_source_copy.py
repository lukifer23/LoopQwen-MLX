"""Preregister the source-mixture fit and one two-system live conversation comparison."""

from pathlib import Path
from collections import defaultdict
from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import read, digest
from loopqwen.qwen_tools import render, declarations
from loopqwen.research.cross_view_study import tokenizer
from scripts.source_copy_material import ROOT, TOKENS, PANEL, SEED, PROVIDER
from scripts.program_language_material import answer, user
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

FIT = "source-copy-fit-01"
PAIR = "source-copy-transfer-01"
PARENT = "bounded-read-fit-01"
RESULT = Path("research/results")
EX = Path("research/experiments")
PROTOCOL = Path("research/diagnostics/source-copy-conversation.json")


def common(w):
    first, second = w["turns"]
    value = first["answers"][0] if w["family"] == "reading" else str(first["reference"])
    h = [
        dict(role="system", content=w["system"]),
        user(w["body"] + "\n\n" + first["question"]),
        answer(value),
    ]
    return h, second["question"]


def run():
    audited = Path("docs/measurements/bounded-read-source-address/summary.json")
    parent = RESULT / PARENT
    sources = {
        str(p): digest(p)
        for p in (
            TOKENS,
            PANEL,
            ROOT / "inventory.json",
            ROOT / "teacher-review.json",
            parent / "checksums.json",
            audited,
        )
    }
    for p in sources:
        committed(Path(p))
    assert read(audited)["copy_investigation"]
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    prompts = []
    for w in unpack(PANEL):
        h, q = common(w)
        defs = declarations([PROVIDER]) if w["family"] == "report" else []
        prompts.append(
            dict(
                id=w["id"],
                family=w["family"],
                first=w["initial"],
                fixed=fingerprint(
                    tok.encode(
                        render(tok, [*h, user(q)], defs), add_special_tokens=False
                    )
                ),
            )
        )
    persist(
        PROTOCOL,
        dict(
            panel=digest(PANEL),
            order=prompts,
            comparison="First question and independent second question with the published first answer have identical frozen prompt IDs. Additionally present the same second user turn following each model's own actual first response and all executed callbacks, preserving every prompt ID and terminal status. Use the same history policy when a model fails to deliver. No correct answer or execution is inserted into the live conversation. Every user request includes its full unchanged source report or passage.",
            scoring="Primary unit is one original source. Require correct delivered answers on both consecutive turns under live histories and on the separate matched second-turn condition. Reading uses independent published open references, case/punctuation/article normalization and full delivered answer, with corroborating masked source review for paraphrases. Financial success requires visible correct numerical quantity, relevant executed operation, correct numeric return and source-appropriate units in masked review on all three presentations. Rounding at displayed precision is accepted within .5 percentage points for percents and max(.01,.1% relative) for other values; source review checks direction and display meaning. Freeze joint two-sided exact paired p <= .05, >=5 net new source successes per family, no lost strict success, no worse completion and all five aggregate costs better. Mask all source judgments before revealing identity. Archive even if gates fail. Preregistered follow-ups require all primary gates and verified native weights: matched-donor ordinary depth, second seed, broad retention, then independent full-context semantic comparison. No other arm in the primary.",
            decode="Both pinned products use official Qwen reasoning-enabled templating, unrestricted greedy, up to 384 tokens per turn, at most three turns per user question, sequential requests, actual callback responses. Report actual wall, first wait, peak native allocation, opening throughput and decoding throughput. Max 4200 native seconds, max 2100 per arm, 3 GiB peak and 1.5 GiB system available. Stop at failed fit or resource limit.",
        ),
    )
    sources[str(PROTOCOL)] = digest(PROTOCOL)
    prior = read(parent / "registration.json")
    keep = {
        k: prior[k]
        for k in (
            "checkpoint",
            "base_text_sha256",
            "config_sha256",
            "donor",
            "passes",
            "shared_query",
            "revisits",
            "rank",
            "gate_rank",
            "compute_precision",
        )
    }
    data = unpack(TOKENS)
    groups = defaultdict(list)
    for w in data:
        groups[w["group"]].append(w)
    probes = [
        w["id"]
        for g, items in sorted(groups.items())
        for w in sorted(items, key=lambda w: fingerprint([SEED, "probe", w["id"]]))[:8]
    ]
    fit = dict(
        **keep,
        id=FIT,
        parent=PARENT,
        content_read=True,
        bounded_write=True,
        source_copy=True,
        fit_seed=SEED,
        training=str(TOKENS),
        sources=sources,
        epochs=3,
        lr=0.00045,
        batch=5,
        chunk=32,
        probes=probes,
        expected_positions=3 * sum(len(w["y"]) for w in data),
        limits=dict(wall_seconds=2200, peak_gib=3, available_gib=1.5),
        question="Does a learned probability mixture between causal source tokens and the vocabulary improve live, source-dependent reading and executable finance conversations?",
        protocol="Keep all bounded parent parameters frozen across two physical reads. Add a context-sensitive copy gate, a contextual-value address initialized from the parent's learned address, and an RMS-conditioned rank-160 residual limited to .25 in normalized language-head coordinates. Learn the normalized mixture probability of every target token under causal teacher forcing, three complete epochs of 900 whole turns including 256 original two-turn human reading answers, 20 source-reviewed teacher action/delivery turns, source-checked calculator actions, result deliveries and earlier protected domains. Equal weight per turn averaged over batches of five. No output grammar, tool shortcut or gold source index at inference. Require 540 finite nonzero optimizer updates, 105498 target exposures, mean probe gain over the frozen parent >.035 nats/token, positive gains in conversational reading, calculator actions and quantity delivery, mean protected reuse/redirect/retention gain >=-.025, copy ablation gain >.005 across reading and actions and positive second physical visit gain. Preserve all original weights exactly. Stop on resource cap; failure closes the held comparison.",
    )
    persist(EX / (FIT + ".json"), fit)
    base = read(EX / "bounded-read-transfer-01.json")
    pair = dict(
        id=PAIR,
        checkpoint=prior["checkpoint"],
        fit_stage=FIT,
        material=str(PANEL),
        sources={**sources, str(EX / (FIT + ".json")): digest(EX / (FIT + ".json"))},
        base_text_sha256=prior["base_text_sha256"],
        max_tokens=384,
        max_turns=3,
        seconds_per_arm=2100,
        limits=dict(wall_seconds=4200, peak_gib=3, available_gib=1.5),
        question=fit["question"],
        protocol=read(PROTOCOL)["scoring"] + " " + read(PROTOCOL)["decode"],
    )
    for k in ("base_revision", "control", "control_model", "checkpoint_revision"):
        if k in base:
            pair[k] = base[k]
    persist(EX / (PAIR + ".json"), pair)


if __name__ == "__main__":
    run()
