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
        inputs_per_second=state["input_tokens"] / seconds,
        positions_per_second=totals("executed_positions") / seconds,
        updates_per_second=len(h) / seconds,
        seconds_per_million_targets=seconds * 1e6 / state["loss_tokens"],
        available_min_bytes=min(r["available"] for r in h),
        initial_swap_bytes=h[0]["swap"],
        final_swap_bytes=h[-1]["swap"],
        max_swap_bytes=max(r["swap"] for r in h),
        peak_rss=max(r["rss"] for r in h),
        mlx_active_max_bytes=max(r["mlx_active"] for r in h),
        mlx_cached_max_bytes=max(r["mlx_cached"] for r in h),
        alpha_initial_forward=h[0]["alpha_forward"],
        last_update={
            k: h[-1][k]
            for k in (
                "exit_ce",
                "raw_exit_ce",
                "alpha_forward",
                "alpha",
                "gate_gradient",
                "gate_lr",
                "lr",
            )
            if k in h[-1]
        },
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


def pass_changes(index):
    result = {}
    for name, left_path in index["records"].items():
        other = name.removesuffix("-k1") + "-k2"
        if not name.endswith("-k1") or other not in index["records"]:
            continue
        left = read(left_path)["scores"]
        right = read(index["records"][other])["scores"]
        for split, run in left.items():
            counts = Counter()
            for a, b, l, r in zip(
                run["records"],
                right[split]["records"],
                run["likelihood"]["records"],
                right[split]["likelihood"]["records"],
                strict=True,
            ):
                if len({v["id"] for v in (a, b, l, r)}) != 1:
                    raise ValueError("Different likelihood and prediction cohorts")
                if b["correct"] and not a["correct"]:
                    counts.update(fixed=1, verified_fixes=int(b.get("verified", False)))
                elif a["correct"] and not b["correct"]:
                    counts.update(
                        broken=1, verified_bypasses_broken=int(a.get("verified", False))
                    )
                if a["prediction"] == b["prediction"]:
                    counts.update(
                        unchanged_answer=1,
                        changed_strings_same_answer=int(a["text"] != b["text"]),
                        increased_conditional_gold_logp=int(
                            r["final_logp"] > l["final_logp"]
                        ),
                    )
            result[f"{other}/{split}"] = dict(n=len(run["records"]), **counts)
    return result


def figures(root, index, seed):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    data = records(index)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    panels = [
        (
            "250K training replication",
            [
                ("dense-250000-k1", "Dense"),
                ("time-250000-k1", "Dense · time"),
                ("gated-250000-k1", "Gated · K1"),
                ("gated-250000-k2", "Gated · K2"),
            ],
        ),
        (
            "100K dual-exit comparison",
            [
                ("dense-100000-k1", "Dense"),
                ("gated-100000-k2", "Gated · K2"),
                ("joint-100000-k1", "Joint · K1"),
                ("joint-100000-k2", "Joint · K2"),
            ],
        ),
    ]
    for ax, (title, arms) in zip(axes, panels):
        for offset, split, color, label in (
            (-0.18, "development", "#1b7866", "ID"),
            (0.18, "development_ood", "#7b5798", "OOD"),
        ):
            scores = [data[k]["scores"][split + "/trace"] for k, _ in arms]
            ax.bar(
                [j + offset for j in range(len(arms))],
                [100 * v["correct"] / v["n"] for v in scores],
                width=0.34,
                color=color,
                label=label,
            )
        ax.set(
            title=title,
            ylabel="Traced answers correct (%)",
            ylim=(0, 100),
            xticks=range(len(arms)),
            xticklabels=[x[1] for x in arms],
        )
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.legend(frameon=False, fontsize=8)
    ax = axes[-1]
    for arm, label, color in (
        ("probe", "Frozen + head", "#798899"),
        ("dense", "Dense + head", "#375982"),
        ("recurrent", "Recurrent + head", "#16785e"),
    ):
        scores = read(OUT / f"state-{seed}-{arm}-install.json")["suite"]
        ax.plot(
            [1, 2, 3],
            [100 * scores[str(n)]["terminal"] / scores[str(n)]["n"] for n in (1, 2, 3)],
            marker="o",
            label=label,
            color=color,
        )
    ax.axhline(12.5, ls=":", color="#777777", label="Uniform guess")
    ax.set(
        title="Pointer map generalization",
        xlabel="Requested transitions",
        ylabel="Terminal state correct (%)",
        ylim=(0, 100),
        xticks=[1, 2, 3],
    )
    ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Phase 4 · independent development seed {seed}")
    fig.savefig(root / "development.png", dpi=160)
    plt.close(fig)


def publish_states(seed, root, adapters):
    """Pair each development record with its exact reproducible categorical head."""
    for arm in ("dense", "probe", "recurrent"):
        run = Path("checkpoints") / f"p4_state_{seed}_{arm}"
        for phase in ("install", "outcome"):
            path = OUT / f"state-{seed}-{arm}-{phase}.json"
            if not path.exists():
                continue
            evaluated = read(path)
            generation = run / read(run / (phase + ".json"))["directory"]
            weight_file = generation / "adapters.safetensors"
            state = read(generation / "state.json")
            if (
                digest(weight_file) != evaluated["weights"]
                or state["identity"] != evaluated["identity"]
            ):
                raise ValueError(
                    f"Published {arm} {phase} head differs from the evaluated checkpoint"
                )
            gz(root / "latent" / (path.stem + ".json.gz"), evaluated)
            gz(root / "curves" / (path.stem + ".json.gz"), state["history"])
            dest = adapters / f"pointer_{arm}_{seed}_{phase}"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(weight_file, dest / weight_file.name)
            save(
                dest / "state.json", {k: v for k, v in state.items() if k != "history"}
            )
            scores = evaluated["suite"]
            accuracy = ", ".join(
                f"{scores[str(n)]['terminal']}/{scores[str(n)]['n']} at {n} hop(s)"
                for n in (1, 2, 3)
            )
            card = [
                f"# Pointer {arm} · {phase} · seed {seed}",
                f"An eight-state categorical readout for the official Qwen3.5-0.8B-Base text decoder at `{state['config']['base_revision']}`. This artifact includes {state['trainable']:,} trained parameters, including the required {state['head_parameters']:,}-weight shared classifier. The final prompt position supplies the readout. The synthetic permutation and start node appear in the prompt. Intermediate and terminal gold states are separate labels.",
                f"Seed {seed} trained for {state['step']} updates on disjoint training permutations. Held-out terminal accuracy was {accuracy}. These measurements concern a symbolic task. [Data and limitations](../../../docs/PHASE4_DATA.md) · [Evaluation](../../../docs/PHASE4_RESULTS.md) · [Base license](../../LICENSE-QWEN)",
                f"Artifact SHA-256 `{evaluated['weights']}`. Training identity `{state['identity']}`. Reproduce with `scripts/run_phase4.sh latent --seed {seed}` after the language experiment has materialized its seed configuration.",
                f"```sh\npython -m loopqwen.training.latent --checkpoint {dest} --map 1 2 3 4 5 6 7 0 --start 2 --hops 3\n```",
            ]
            (dest / "README.md").write_text("\n\n".join(card) + "\n")


def publish_stock(seed, adapters):
    source = Path("exports") / f"p4_stock_{seed}"
    measured = read(OUT / "deployment.json")
    if digest(source / "adapters.safetensors") != measured["weights"]:
        raise ValueError("Stock adapter differs from deployment audit")
    dest = adapters / f"stock_gated_{seed}_k1"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("adapters.safetensors", "adapter_config.json", "deployment.json"):
        shutil.copyfile(source / name, dest / name)
    (dest / "README.md").write_text(
        f"# Stock MLX-LM K=1 deployment\n\n"
        f"The gated 250K seed-{seed} adapter was trained with two executions of layers 10–15. "
        f"This format runs the ordinary 24-layer decoder with stock MLX-LM 0.32.2. "
        f"The base revision is `{measured['revision']}`. The isolated loader matched "
        f"all greedy tokens on {len(measured['validation'])} traced validation prompts. "
        f"SHA-256 `{measured['weights']}`. [Training card](../gated_{seed}_250k/README.md) · [License](../../LICENSE-QWEN)\n\n"
        f"```python\nfrom huggingface_hub import snapshot_download\nfrom mlx_lm import load\n"
        f"base = snapshot_download({measured['base']!r}, revision={measured['revision']!r})\n"
        f"model, tokenizer = load(base, adapter_path={str(dest)!r})\n```\n"
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
        "data-audit.json",
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
        save(root / (label + "-pass-changes.json"), pass_changes(idx))
    all_handles = {}
    for idx in (index, *stages.values()):
        all_handles.update(idx["records"])
    for path in sorted((OUT / "configs").glob("p4_profile_*.yaml")):
        state = read(
            checkpoint_location(Path("checkpoints") / path.stem) / "state.json"
        )
        gz(
            root / "profiles" / (path.stem + ".json.gz"),
            dict(
                git=state["git"],
                identity=state["training_identity"],
                order=state["order_hash"],
                history=state["history"],
            ),
        )
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
    publish_states(seed, root, adapters)
    publish_stock(seed, adapters)
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
    figures(root, index, seed)
    save(
        root / "adapter-checksums.json",
        {
            str(p.relative_to(adapters)): digest(p)
            for p in sorted(adapters.rglob("*"))
            if p.is_file()
        },
    )
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
