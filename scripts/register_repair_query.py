"""Reserve the original paper cases and external tool histories for one source-repair comparison."""

import argparse
from pathlib import Path

from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from scripts.provider_field_mutations import LOCAL, OUTPUT
from scripts.provider_recovery_field import PANEL as DEV
from scripts.register_shared_query import PANEL as SHORT
from scripts.repair_query_material import ROOT
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

EX = Path("research/experiments")
RESULT = Path("research/results")
DOC = Path("docs/measurements")
FEATURES = "repair-query-features-01"
FIT = "repair-query-fit-01"
PRIMARY = "repair-query-transfer-01"
PANEL = ROOT / "transfer-panel.json.gz"
PLAN = Path("research/diagnostics/repair-query-reading.json")
PRECISION = DOC / "shared-query-precision-idle.json"


def panel():
    rows = list(unpack(SHORT)["rows"])
    notes = {r["id"]: r for r in read(DEV.with_name("field-wording.json"))["items"]}
    for path in (DEV, LOCAL):
        for w in unpack(path)["rows"]:
            ref = w["id"].rsplit("/", 1)[0] + "/unassisted"
            reading = notes[ref]
            rows.append(
                {
                    **w,
                    "family": "field",
                    "questions": [w["question"]],
                    "initial_sha256": w["prompt_sha256"],
                    "defs": w["tools"],
                    "cluster": w["source"],
                    "reference": w["human"],
                    "interpretation": reading,
                    "source_split": "official_development"
                    if path == DEV
                    else "locally_withheld_training",
                }
            )
    assert len(rows) == len({w["id"] for w in rows}) == 180
    rows.sort(key=lambda r: fingerprint([864371, "repair field", r["id"]]))
    source = {
        str(p): digest(p)
        for p in (
            SHORT,
            DEV,
            LOCAL,
            OUTPUT,
            DEV.with_name("field-denotations.json"),
            DEV.with_name("field-wording.json"),
            DEV.with_name("field-score-controls.json"),
        )
    }
    result = {"rows": rows, "sources": source}
    if PANEL.exists():
        assert unpack(PANEL) == result
    else:
        packed(PANEL, result)
    persist(
        PLAN,
        {
            "material": digest(PANEL),
            "ordered": [w["id"] for w in rows if w["family"] != "locale"],
            "scope": "All 176 source responses, masked across arms before revealing quality or costs. A single project reviewer rates unique full visible responses and their actual calls on every item.",
            "rating": "Strategy requires a supported conclusion and explanation; finance requires a correct amount with units and an actual operative calculation. For database questions evaluate the original user intent, original source cells and entire supplied history. Credit a warranted correction of a faulty premise or alternate entity interpretation when grounded in supplied evidence or an actual successful query. On an incorrect prior result or error a fresh relevant tool execution is required. Review the seven preflagged ambiguous questions separately; on unflagged questions require the executable source-equivalence check for full grounded credit.",
            "gate": "At least six net new fully usable responses on EACH of strategy, report and database requests, no individual losses, exact paired sign-flip p<=0.05 within each, and no increase in unsupported answers. Combine all four conditions and both questions within each database before the field randomization test. Require positive net usable gains on each of the four field conditions. Preserve the separate strict automatic decision, all cases and each original source split.",
        },
    )
    print(
        "Frozen 180 requests on 12 database schemas, 32 reports, 48 human reasoning topics, and four complete long files"
    )


def eligible():
    for name in ("elastic-world-fit-01", "elastic-world-transfer-01"):
        root = RESULT / name
        committed(root / "checksums.json")
        if name.endswith("transfer-01") and (root / "control.json.gz").is_file():
            for group in ("services", "qualitative"):
                committed(DOC / ("elastic-" + group + "-reading.json"))
            if not read(DOC / "elastic-services-reading.json")["passed"]:
                return {
                    "stage": name,
                    "archive": digest(root / "checksums.json"),
                    "source_reading": digest(DOC / "elastic-services-reading.json"),
                }
        if not read(root / "decision.json")["passed"]:
            return {"stage": name, "archive": digest(root / "checksums.json")}
    raise ValueError("The qualifying predecessor requires replication")


def register():
    prior = eligible()
    committed(PRECISION)
    rc = read(PRECISION)
    if not rc["passed"] or len(rc["records"]) != 12:
        raise ValueError("A completed passing precision audit is required")
    committed(PANEL)
    committed(PLAN)
    inv = read(ROOT / "verified-inventory.json")
    original = read(EX / "shared-query-features-01.json")
    common = {
        k: original[k]
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
            "fit_seed",
        )
    }
    sources = {
        str(p): digest(p)
        for p in (
            ROOT / "verified-inventory.json",
            ROOT / "verified-training.json.gz",
            ROOT / "verified-evidence.json.gz",
            PANEL,
            PLAN,
            PRECISION,
            RESULT / prior["stage"] / "checksums.json",
            DEV.with_name("field-score-controls.json"),
            ROOT / "panel-replay.json",
        )
    }
    common.update(
        compute_precision="float16",
        training=str(ROOT / "verified-training.json.gz"),
        evidence=str(ROOT / "verified-evidence.json.gz"),
        sources=sources,
        precision_receipt=str(PRECISION),
        predecessor=prior,
        cold_groups=[
            "strategy",
            "report_action",
            "full_rows",
            "repair_after_schema",
            "redirect_valid_read",
        ],
        limits={"wall_seconds": 4380, "peak_gib": 3.0, "available_gib": 1.5},
    )
    feat = {
        **common,
        "id": FEATURES,
        "max_pressure_restarts": 6,
        "examples": inv["examples"],
        "opening_positions": inv["opening_positions"],
        "target_positions": inv["target_positions"],
        "question": "Can all source-dependent training examples retain the complete ordinary Qwen attention keys and values and true tied-read parity in 16-bit arithmetic?",
        "protocol": original["protocol"]
        + " Use the independently qualified FP16 donor computation. Retain all source positions, token labels, six causal cache offsets, and the original 0.55 top-logit tolerance. Require complete physical extraction. Release the worker on each memory-pressure warning and journal every completed file. Resume only after ten nominal seconds, at most six restarts within the same 4380 charged native seconds and two-hour session. A failed prerequisite closes subsequent fits.",
    }
    persist(EX / (FEATURES + ".json"), feat)
    fit = {
        **common,
        "id": FIT,
        "feature_stage": FEATURES,
        "max_pressure_restarts": 6,
        "checkpoint_interval": 16,
        "epochs": 2,
        "lr": 0.00022,
        "teacher_kl": 0.24,
        "evidence_weight": 8.0,
        "micro_positions": 32,
        "update_positions": 512,
        "expected_positions": inv["target_positions"] * 2,
        "expected_citations": inv["human_cited_queries"] * 2,
        "repair_groups": [
            "repair_after_schema",
            "repair_known_schema",
            "repair_column_roles",
            "redirect_valid_read",
            "answer_valid_read",
        ],
        "question": "Can gradients through two actual shared attention reads improve held-in source reasoning, financial tool actions, and responses to real or misleading provider observations while preserving earlier capabilities?",
        "protocol": "Exactly two deterministic epochs of 6394 distinct full targets. Every cited source range remains token-bound. Require all finite nonzero gradients, group-balanced NLL gain >=0.04 nats, positive gain in strategy, report_action and all five targeted repair groups, protected instruction/explanation/arithmetic/legacy means no worse than -0.02, mean second versus first visit >0.002 nats at fitted weights and citation mass gain >0.02 nats. Hash every upstream feature and independently restore the fitted checkpoint before entry to the direct study. Release the worker and preserve optimizer state on host pressure. Allow at most six restarts from a complete optimizer update after ten nominal seconds, with all workers sharing 4380 native seconds and a two-hour attendance window. Stop permanently at resource cap or failed quality. Data and targets are identical across comparisons requiring a matched donor.",
    }
    fit["sources"] = {
        **sources,
        str(EX / (FEATURES + ".json")): digest(EX / (FEATURES + ".json")),
    }
    persist(EX / (FIT + ".json"), fit)
    primary = {
        **common,
        "id": PRIMARY,
        "fit_stage": FIT,
        "material": str(PANEL),
        "field_history": True,
        "max_tokens": 384,
        "max_turns": 3,
        "context_turns": 1,
        "limits": {"wall_seconds": 4380, "peak_gib": 8.25, "available_gib": 1.5},
        "question": "Does one physically recurrent model improve credible database repairs, expert calculations, multi-step explanations and complete 128K-plus requests over the untouched ordinary Base?",
        "protocol": "One 180-request matched direct pair. Both arms use identical native Qwen thinking prompts, unconstrained greedy generation, source histories, real callbacks and output caps. Record time, first-token wait, peak, prefill and decode alongside all successes. Require >=6 net improvements and zero lost successes separately in database, strategy and financial families with two-sided exact p<=0.05. For database cases use schema-cluster sign flips and positive gains under each of the four conditions. Each short family requires at least as many completions. Four unabridged prompts above 131072 tokens must all be correct and strictly exceed Base. Require lower total wall, first-token wait and peak and faster opening and decode. Grade all 176 masked short cases before examining the cost or decision. A joint pass is prerequisite to the fully matched ordinary-depth training, second independent seed, broad retention and clean 128K deployment. All failed registered prerequisites close downstream stages.",
    }
    primary["sources"] = {
        **sources,
        str(EX / (FIT + ".json")): digest(EX / (FIT + ".json")),
    }
    persist(EX / (PRIMARY + ".json"), primary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("panel", "register"))
    if parser.parse_args().mode == "panel":
        panel()
    else:
        register()
