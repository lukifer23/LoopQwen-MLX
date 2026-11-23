"""Archive completed Phase 4 studies and render tables from immutable measurements."""

import argparse
import gzip
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loopqwen.io import digest, read, save, checkpoint_location
from loopqwen.identity import fingerprint

OUT = Path("outputs/phase4")


def table(header, rows):
    return "\n".join(
        ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def gz(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        gzip.compress(json.dumps(data, separators=(",", ":")).encode(), mtime=0)
    )


def training(state):
    h = state["history"]
    seconds = sum(r["seconds"] for r in h)
    totals = lambda key: sum(r[key] for r in h)
    by_source = Counter()
    for r in h:
        by_source.update(r["scored_by_source"])
    return dict(
        step=state["step"],
        labels=state["loss_tokens"],
        inputs=state["input_tokens"],
        examples=sum(state["visits"].values()),
        unique=len(state["visits"]),
        order=state["order_hash"],
        seconds=seconds,
        tps=state["loss_tokens"] / seconds,
        peak=max(r["peak_bytes"] for r in h) / 2**30,
        padding=1 - state["input_tokens"] / totals("executed_positions"),
        source=by_source,
        validation_seconds=sum(r.get("validation_seconds", 0) for r in h),
        positions=totals("executed_positions"),
        exit_targets=totals("exit_targets"),
        site_positions=totals("decoder_site_positions"),
        alpha=h[-1]["alpha"],
        grad_norm=h[-1]["gradient_norm"],
        config=state["config"],
        identity=state["training_identity"],
        capacity=state["trainable"],
    )


def records(index):
    return {k: read(v) for k, v in index["records"].items()}


def eligibility(index):
    rec = records(index)

    def scores(name):
        r = rec[name]
        tr = [r["scores"][s + "/trace"] for s in ("development", "development_ood")]
        return (
            sum(v["correct"] for v in tr) / sum(v["n"] for v in tr),
            [v["verified"] for v in tr],
            r["retention"]["nll"],
        )

    def passes(name, control, delta):
        a, va, la = scores(name)
        b, vb, lb = scores(control)
        return (
            a - b >= delta - 1e-10
            and all(x > y for x, y in zip(va, vb))
            and la <= lb + 0.05
        )

    return dict(
        track_a=all(
            passes("gated-250000-k1", s + "-250000-k1", 0.03) for s in ("dense", "time")
        ),
        track_b=passes("joint-100000-k2", "joint-100000-k1", 0.03)
        and scores("joint-100000-k2")[0] >= scores("dense-100000-k1")[0],
    )


def publish(seed):
    index = read(OUT / f"evaluation-{seed}.json")
    stages = {
        key: read(OUT / f"evaluation-{seed}-{key}.json")
        for key in ("boundary", "v4")
        if (OUT / f"evaluation-{seed}-{key}.json").exists()
    }
    root = Path("docs/measurements/phase4")
    reports = Path("docs")
    adapters = Path("adapters/phase4")
    if (root / "checksums.json").exists():
        base = OUT / "reproductions" / fingerprint(index)[:12]
        reports = base / "docs"
        root = reports / "measurements/phase4"
        adapters = base / "adapters/phase4"
    root.mkdir(exist_ok=True, parents=True)
    for name in (
        "profiles.json",
        "execution.json",
        "gradient-audit.json",
        "gradient-exits.json",
        "gradient-bf16.json",
        "regression.json",
        "deployment.json",
        "merged-deployment.json",
        "exposure.json",
    ):
        if (OUT / name).exists():
            shutil.copyfile(OUT / name, root / name)
    for ds in ("v2", "v3", "v4", "v4_pointer"):
        shutil.copyfile(
            Path("data") / ds / "manifest.json", root / (ds + "-manifest.json")
        )
    metric_tables = {}
    for label, idx in {"primary": index, **stages}.items():
        block = []
        for arm, path in idx["records"].items():
            rec = read(path)
            gz(root / "eval" / label / (arm + ".json.gz"), rec)
            scores = list(rec["scores"].values())
            direct = scores[::2]
            traced = scores[1::2]
            block.append(
                [
                    arm,
                    *[f"{r['correct']}/{r['n']}" for r in traced],
                    *[r["verified"] for r in traced],
                    *[r["correct"] for r in direct],
                    f"{rec['retention']['nll']:.4f}",
                    f"{traced[0]['interactive']['seconds'] / 8:.3f}",
                    f"{sum(r['n'] for r in traced) / sum(r['seconds'] for r in traced):.2f}",
                ]
            )
        metric_tables[label] = table(
            [
                "Arm · targets · depth",
                "ID traced",
                "OOD traced",
                "ID verified",
                "OOD verified",
                "ID direct",
                "OOD direct",
                "Language NLL",
                "Interactive s/request",
                "Batched traced requests/s",
            ],
            block,
        )
        save(root / (label + "-pairs.json"), idx["pairs"])
    all_handles = {}
    for idx in (index, *stages.values()):
        all_handles.update(idx["records"])
    metrics = {}
    for name, path in all_handles.items():
        arm, budget, depth = name.rsplit("-", 2)
        if arm == "base" or depth != "k1":
            continue
        ck = Path("checkpoints") / f"p4_{seed}_{'dense' if arm == 'time' else arm}"
        if arm == "dense" and not (ck / f"targets-{budget}.json").exists():
            ck = Path("checkpoints") / f"p4_{seed}_dense_data"
        gen = (
            checkpoint_location(ck)
            if arm == "time"
            else ck / read(ck / f"targets-{budget}.json")["directory"]
        )
        state = read(gen / "state.json")
        exp = read(path)
        if digest(gen / "adapters.safetensors") != exp["identity"]["weights"]:
            raise ValueError("Evaluation weights differ " + name)
        metrics[name] = training(state)
        run = read(OUT / "runs" / (ck.name + ".json"))
        metrics[name]["run_wall_s"] = run["seconds"]
        metrics[name]["git"] = state["git"]
        metrics[name]["weights"] = (
            run["weights"] if arm == "time" else digest(gen / "adapters.safetensors")
        )
        io = ck / "serialization.jsonl"
        records_io = (
            [json.loads(line) for line in io.open() if line.strip()]
            if io.exists()
            else []
        )
        metrics[name]["serialization"] = [
            z for z in records_io if z["step"] <= state["step"]
        ]
        gz(root / "curves" / (name + ".json.gz"), state["history"])
        if int(budget) == 250000 or (arm == "joint" and int(budget) == 100000):
            dest = adapters / (
                f"time_{seed}_matched_seconds"
                if arm == "time"
                else f"{arm}_{seed}_{int(budget) // 1000}k"
            )
            dest.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(gen / "adapters.safetensors", dest / "adapters.safetensors")
            save(
                dest / "state.json",
                {k: v for k, v in state.items() if k not in ("history", "visits")},
            )
            k = state["config"]["passes"]
            s = exp["scores"]
            (dest / "README.md").write_text(f"""# {arm} seed {seed}

Experimental unmerged LoRA for Qwen/Qwen3.5-0.8B-Base, revision `{state["config"]["base_revision"]}`. See the Qwen license in the parent adapters tree. Seed {seed} saw {state["loss_tokens"]:,} supervised V2 tokens over {state["step"]} updates. Training depth K={k}, first-exit objective coefficient {state["config"].get("first_exit_weight", 0)} and {state["trainable"]:,} trainable parameters. The scalar gate uses sigmoid interpolation whenever active. K=1 deployment executes all 24 original layers once. V3 development K=1 solved {s["development/trace"]["correct"]}/150 ID and {s["development_ood/trace"]["correct"]}/150 OOD. Single-seed development results constrain claims of generalization.

SHA-256 `{digest(dest / "adapters.safetensors")}`. Use the locked native environment and `python -m loopqwen generate --checkpoint {dest} --passes 1 --prompt 'Evaluate 2 + 3.' --max-tokens 32`. Reproduce with `scripts/run_phase4.sh {"dual-exit" if arm == "joint" else "replicate"} --seed {seed}` after the efficiency stage. Full evaluation and latency measurements are in the [Phase 4 report](../../../docs/PHASE4_RESULTS.md).
""")
    breakdown = {}
    for suite, idx in {"primary": index, **stages}.items():
        for name, path in idx["records"].items():
            for split, score in read(path)["scores"].items():
                groups = {}
                for r in score["records"]:
                    for key in (
                        r["category"],
                        f"complexity/{r['complexity']}",
                        f"steps/{r['steps']}",
                    ):
                        bucket = groups.setdefault(key, Counter())
                        bucket.update(
                            examples=1,
                            correct=int(r["correct"]),
                            verified=int(r.get("verified", False)),
                            truncated=int(r["token_cap_reached"]),
                            generated=r["tokens"],
                        )
                breakdown[f"{suite}/{name}/{split}"] = groups
    gz(root / "difficulty.json.gz", breakdown)
    for path in sorted((OUT / "configs").glob("*.yaml")):
        directory = root / "configs"
        directory.mkdir(exist_ok=True)
        shutil.copyfile(path, directory / path.name)
    for path in sorted(OUT.glob("p4_*.schedule.gz")):
        # Identical dense/gated schedules share one content-addressed copy.
        directory = root / "schedules"
        directory.mkdir(exist_ok=True)
        import hashlib

        key = hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest()
        dest = directory / (key + ".jsonl.gz")
        if not dest.exists():
            shutil.copyfile(path, dest)
    save(root / "training.json", metrics)
    for label in ("dense", "probe", "recurrent"):
        for stage in ("install", "outcome"):
            file = OUT / f"state-{seed}-{label}-{stage}.json"
            if file.exists():
                gz(root / "latent" / (file.stem + ".json.gz"), read(file))
    latent = []
    for file in sorted((root / "latent").glob("*.gz")):
        x = json.loads(gzip.decompress(file.read_bytes()))
        latent.append(
            (
                file.stem.removesuffix(".json"),
                x["step"],
                x["readout"],
                *[f"{s['terminal']}/{s['n']}" for s in x["suite"].values()],
                x["suite"]["3"]["first_terminal"],
                x["eligible"],
            )
        )
    result = [
        "# Phase 4 research results",
        "",
        f"Exploratory seed {seed} on M3 Pro, starting at `0a0bb9c`. Development evaluation used the complete 150+150 V3 sets at 100K and 250K, and the first 30+30 at 25K. Trace generation uses 192 tokens and direct generation uses 24. Every trained comparison has a true single-pass exit. Phase 3 sealed cohorts were excluded. [Registration](PHASE4_PLAN.md)",
        "",
        metric_tables["primary"],
        "",
        "## Paired primary outcomes",
        "",
    ]
    pairs = []
    for k, v in index["pairs"].items():
        if "250000" not in k and "100000" not in k:
            continue
        for split, r in v.items():
            if not split.endswith("trace"):
                continue
            pairs.append(
                (
                    k,
                    split,
                    r["wins"],
                    r["losses"],
                    r["verified_change"],
                    f"{100 * r['difference']:+.1f}",
                    f"[{100 * r['item_ci95'][0]:.1f}, {100 * r['item_ci95'][1]:.1f}]",
                    f"{r['mcnemar']:.3f}",
                )
            )
    result += [
        table(
            [
                "Reference / challenger",
                "Split",
                "Fixes",
                "Breaks",
                "Verified Δ",
                "Δ pp",
                "Paired item 95%",
                "Exact p",
            ],
            pairs,
        ),
        "",
        "## Categorical pointer prediction",
        "",
        table(
            [
                "Arm / phase",
                "Updates",
                "Head parameters",
                "1 hop",
                "2 hops",
                "3 hops",
                "3-hop terminal at pass 1",
                "Eligible for annealing",
            ],
            latent,
        ),
    ]
    for label in stages:
        result += [
            "",
            "## " + label.upper() + " exploratory comparison",
            "",
            metric_tables[label],
        ]
    decisions = eligibility(index)
    result += [
        "",
        "## Registered decisions",
        "",
        f"Track A eligibility: {decisions['track_a']}. Track B eligibility: {decisions['track_b']}. Confidence intervals above resample items for this one seed. [Training cost and memory](PHASE4_OPTIMIZATIONS.md) · [Data and verification](PHASE4_DATA.md)",
    ]
    (reports / "PHASE4_RESULTS.md").write_text("\n".join(result) + "\n")
    profiles = read(root / "profiles.json")
    perf = [
        "# Phase 4 native performance",
        "",
        "MLX 0.32.4 / MLX-LM 0.32.2, M3 Pro 14-core GPU, 18 GiB unified memory. The configuration screen used the same ten seed-109 logical updates (first two warmup). Sequence cap 256, batch four, BF16, gated K=2 with block rank eight. Timers include evaluated optimizer updates. The workload and the measured memory envelope selected 360 padded positions without activation recomputation.",
        "",
        table(
            [
                "Cap",
                "Checkpointing",
                "Targets/s",
                "Useful inputs/s",
                "Positions/s",
                "Median update s",
                "Range s",
                "Peak GiB",
            ],
            [
                (
                    r["cap"],
                    r["policy"],
                    round(r["tps"], 1),
                    round(r["inputs_s"], 1),
                    round(r["positions_s"], 1),
                    round(r["median_s"], 3),
                    f"{min(r['update_times']):.3f}–{max(r['update_times']):.3f}",
                    round(r["peak"] / 2**30, 2),
                )
                for r in profiles
            ],
        ),
        "",
        "## Sustained production runs",
        "",
        table(
            [
                "Arm / milestone",
                "Updates",
                "Targets",
                "Unique",
                "Update seconds",
                "Targets/s",
                "Padding",
                "Peak GiB",
                "Validation s",
                "Checkpoint s",
                "Full invocation wall s¹",
            ],
            [
                (
                    k,
                    r["step"],
                    r["labels"],
                    r["unique"],
                    round(r["seconds"], 1),
                    round(r["tps"], 1),
                    f"{r['padding']:.1%}",
                    round(r["peak"], 2),
                    round(r["validation_seconds"], 1),
                    round(sum(x["seconds"] for x in r["serialization"]), 2),
                    round(r["run_wall_s"], 1),
                )
                for k, r in metrics.items()
            ],
        ),
        "",
        "¹ Every milestone on a trajectory has the same full invocation wall time. The cumulative update, validation and checkpoint columns stop at the named milestone. Small additional costs include parsing, startup, per-update telemetry, serialization of the separate output summary, and finalization.",
        "",
        "[Source-resolved configurations, work counters and checkpoint measurements](measurements/phase4/training.json)",
    ]
    (reports / "PHASE4_OPTIMIZATIONS.md").write_text("\n".join(perf) + "\n")
    save(
        root / "checksums.json",
        {
            str(f.relative_to(root)): digest(f)
            for f in sorted(root.rglob("*"))
            if f.is_file() and f.name != "checksums.json"
        },
    )
    return reports / "PHASE4_RESULTS.md"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=109)
    print(publish(p.parse_args().seed))
