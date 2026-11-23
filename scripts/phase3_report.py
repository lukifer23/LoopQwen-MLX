"""Render publication tables and a learning curve from completed local measurements."""

import argparse
import gzip
import json
from statistics import median
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loopqwen.io import digest, read, save


def table(head, data):
    return "\n".join(
        ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        + ["| " + " | ".join(map(str, row)) + " |" for row in data]
    )


def docs(archive, destination):
    dest = Path(destination)
    dest.mkdir(exist_ok=True, parents=True)
    train = read(archive / "training.json")
    main = read(archive / "main.json")
    screen = read(archive / "screen.json")
    chosen = read(archive / "selection.json")["selected"]
    transfer = read(archive / "fresh-development.json")
    curves = read(archive / "learning-curves.json")
    data = read(archive / "exposure.json")["100000"]
    post = read(archive / "answer-likelihood.json")
    rows = main["rows"]
    arms = ["dense", "direct", chosen]

    def quality(data):
        return table(
            [
                "Arm",
                "Execution",
                "Answers",
                "Certificates",
                "Capped",
                "Final NLL",
                "Trace NLL",
                "Prose NLL",
                "Greedy seconds",
            ],
            [
                (
                    r["arm"],
                    r["mode"],
                    f"{r['correct']}/{r['n']}",
                    r["verified"],
                    r["truncated"],
                    f"{r['final_nll']:.5f}",
                    f"{r['trace_nll']:.4f}",
                    f"{r['retention']:.4f}",
                    f"{r['seconds']:.1f}",
                )
                for r in data
            ],
        )

    selected = next(r for r in rows if r["arm"] == chosen and r["mode"] == "trained")
    dense = next(r for r in rows if r["arm"] == "dense")
    bypass = next(r for r in rows if r["arm"] == chosen and r["mode"] == "bypass")
    promote = (
        selected["correct"] > bypass["correct"]
        and selected["correct"] >= dense["correct"]
        and selected["retention"] <= dense["retention"] + 0.05
    )
    report = [
        "# Phase 3 results",
        "",
        "One development seed tested whether a near-identity second block application could earn its extra cost after substantially more supervised training. The comparison starts from Qwen/Qwen3.5-0.8B-Base revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. All three 100K arms used native Metal, BF16, 752,393,024 shared pretrained text parameters and 1,336,320 adapter parameters. The learned gate adds one scalar. [Registration](PHASE3_PLAN.md)",
        "",
        f"The dense, direct and {chosen} arms reached {data['targets']:,} supervised tokens at update {data['step']}. Each consumed {data['inputs']:,} useful input tokens and {sum(data['examples'].values()):,} distinct problems, with {data['examples']['answer']} answer and {data['examples']['trace']} trace responses. The realized target partition was {data['classes']['format']:,} formatting, {data['classes']['intermediate']:,} intermediate, {data['classes']['final']:,} final and {data['classes']['eos']:,} EOS tokens. The Phase 2 confirmatory arms saw 336 examples. The development examples, decoding cap and scoring protocol differ between studies.",
        "",
        "## Gate learning and model selection",
        "",
        "LR 0.001 and 0.01 both updated the gate. Their eight-update trace NLLs were %.6f and %.6f. The registered tie breakers selected %s at 25K. This selection used development data."
        % (
            read(archive / "learn1-lr.json")["loss"]["nll"],
            read(archive / "learn2-lr.json")["loss"]["nll"],
            chosen,
        ),
        "",
        quality(screen["rows"]),
        "",
        "## Main development comparison",
        "",
        quality(rows),
        "",
        "Final NLL here conditions on the correct intermediate trace. V2 traces generally repeat the answer from the preceding intermediate line. The answer-only NLL below scores the response directly after the problem.",
        "",
        table(
            [
                "Arm",
                "K",
                "V2 answer-only NLL",
                "V3 OOD answer-only NLL",
                "V3 retention NLL",
            ],
            [
                (
                    k.split("/")[0],
                    k.split("/")[1],
                    f"{r['answer_only']['v2']['nll']:.4f}",
                    f"{r['answer_only']['v3_ood']['nll']:.4f}",
                    f"{r['v3_retention']['nll']:.4f}",
                )
                for k, r in post.items()
            ],
        ),
        "",
        "Paired differences below compare the trained execution with the named comparator. Bootstrap intervals reflect item variability within this development cohort. Training-seed uncertainty remains unmeasured.",
        "",
        table(
            [
                "Pair",
                "Fixed",
                "Broken",
                "Difference (pp)",
                "Item 95% interval (pp)",
                "Exact McNemar p",
            ],
            [
                (
                    k,
                    v["fixes"],
                    v["breaks"],
                    f"{100 * v['delta']:.1f}",
                    f"[{100 * v['item_ci95'][0]:.1f}, {100 * v['item_ci95'][1]:.1f}]",
                    f"{v['mcnemar']:.3f}",
                )
                for k, v in main["paired"].items()
            ],
        ),
        "",
        "## Fresh development examples",
        "",
        "These larger sets include genuinely branching DAGs. The independent retention evaluation covers five documents. The sealed confirmation splits remain available.",
        "",
        table(
            [
                "Development split",
                "Arm/K",
                "Answers",
                "Certificates",
                "Capped",
                "Final NLL",
                "Seconds",
            ],
            [
                (
                    split,
                    k,
                    f"{r['correct']}/{r['n']}",
                    r["verified"],
                    r["truncated"],
                    f"{r['final_nll']:.4f}",
                    f"{r['seconds']:.1f}",
                )
                for split, s in transfer.items()
                for k, r in s["metrics"].items()
            ],
        ),
        "",
        "## Learning curves",
        "",
        "![Development learning curves](measurements/phase3/learning-curves.png)",
        "",
        "## Promotion decision",
        "",
        f"The preregistered 100K promotion rule {'passed' if promote else 'did not pass'}. Selected K=2 scored {selected['correct']}/{selected['n']}, its own bypass scored {bypass['correct']}/{bypass['n']} and dense scored {dense['correct']}/{dense['n']}. The selected-minus-dense literary NLL difference was {selected['retention'] - dense['retention']:+.4f}. These are exploratory measurements.",
        "",
        "[Training, memory and deployment measurements](PHASE3_OPTIMIZATIONS.md) · [Run identifiers and artifacts](EXPERIMENT_REGISTRY.md)",
    ]
    curve_file = archive / "curves" / ("p3_" + chosen + "_250k.json.gz")
    if not curve_file.exists():
        curve_file = archive / "curves" / ("p3_" + chosen + ".json.gz")
    history = json.loads(gzip.decompress(curve_file.read_bytes()))
    gate_points = []
    total_targets = 0
    for rec in history:
        total_targets += rec["labels"]
        if "validation" in rec and "gate_logit" in rec:
            gate_points.append(
                (
                    rec["step"],
                    total_targets,
                    f"{rec['alpha']:.6f}",
                    f"{rec['gate_logit']:.5f}",
                    f"{rec['gate_gradient']:.3g}",
                    f"{rec['gate_delta']:+.3g}",
                    f"{rec['gate_lr']:.3g}",
                    f"{rec['lr']:.3g}",
                )
            )
    if gate_points:
        report += [
            "",
            "## Effective gate trajectory",
            "",
            "The initial alpha was 0.10 and its sigmoid logit was −2.19722. Each listed gradient is token-normalized before joint clipping. Delta is the logit change after the actual AdamW update.",
            "",
            table(
                [
                    "Step",
                    "Targets",
                    "Alpha",
                    "Logit",
                    "Gate gradient",
                    "Logit delta",
                    "Gate LR",
                    "Adapter LR",
                ],
                gate_points,
            ),
        ]
    if (archive / "promotion.json").exists():
        extended = read(archive / "promotion.json")
        report += [
            "",
            "## Promoted 250K pair",
            "",
            "The two arms continued their original AdamW state on the same planned examples, with per-layer checkpointing throughout the extension. Each reached update 1,605 with 250,031 targets, 597,444 input tokens and 6,420 distinct instances. Fresh development scores below remain single-seed observations.",
            "",
            table(
                [
                    "Cohort",
                    "Arm/execution",
                    "Answers",
                    "Certificates",
                    "Capped",
                    "Final-token NLL",
                    "Retention NLL",
                    "Seconds",
                ],
                [
                    (
                        split,
                        k,
                        f"{r['correct']}/{r['n']}",
                        r["verified"],
                        r["truncated"],
                        f"{r['final_nll']:.5f}",
                        f"{r['retention']:.4f}",
                        f"{r['seconds']:.1f}",
                    )
                    for split, record in extended["splits"].items()
                    for k, r in record["metrics"].items()
                ],
            ),
            "",
            "The registered eligibility rule for a three-seed confirmation "
            + (
                "passed."
                if extended["confirm_eligible"]
                else "did not pass. The fresh confirmatory cohorts stay sealed."
            ),
        ]
        report += [
            "",
            table(
                [
                    "Fresh split",
                    "Gated K=2 versus",
                    "Fixes",
                    "Breaks",
                    "Difference (pp)",
                    "Paired item 95% (pp)",
                    "Exact McNemar p",
                ],
                [
                    (
                        split,
                        comparator,
                        pair["fixes"],
                        pair["breaks"],
                        f"{100 * pair['delta']:+.1f}",
                        f"[{100 * pair['item_ci95'][0]:.1f}, {100 * pair['item_ci95'][1]:.1f}]",
                        f"{pair['mcnemar']:.3f}",
                    )
                    for split, record in extended["splits"].items()
                    if split != "v2_validation"
                    for comparator, pair in (
                        ("matched dense", record["against_dense"]),
                        ("same checkpoint K=1", record["against_bypass"]),
                    )
                ],
            ),
        ]
        from scripts.phase3_changes import changes

        flips = changes(archive, chosen)
        report.extend(
            [
                "",
                "### Problems changed by the extra pass",
                "",
                table(
                    [
                        "Fresh cohort",
                        "Problem ID",
                        "Family",
                        "K=1 prediction",
                        "K=2 prediction",
                        "Gold",
                        "Effect",
                        "K=2 trace verified",
                    ],
                    [
                        (
                            r["split"],
                            r["id"][:12],
                            r["family"],
                            r["bypass"]["prediction"],
                            r["gated"]["prediction"],
                            r["gold"],
                            r["effect"],
                            r["gated"]["verification"]["verified"],
                        )
                        for r in flips
                    ],
                ),
                "",
                "Both final-answer fixes contained incorrect intermediate arithmetic. In the modular chain, the second pass wrote 39 mod 17 = 12. In the linear system it evaluated 132 − 143 as 13. Both regressions occurred on left rotations by three positions. In each, K=1 correctly rotated the array and selected the alternating indices, while K=2 rotated by two. These four changes identify specific failure modes on this seed.",
            ]
        )
        if not extended["confirm_eligible"]:
            report += [
                "",
                "On both fresh cohorts the learned checkpoint retained its higher observed score over dense when decoded at K=1. The extra pass fixed one bypass error and broke one correct answer on each cohort. The item intervals describe this seed. They cannot estimate variation across independent training runs.",
                "",
                "The next experiment should replicate dense training versus gated training with both checkpoints decoded at K=1 across independent seeds on development data. Include a dense arm trained for the same measured time, charging its extra training examples. This isolates the possible training-time benefit at equal deployed depth before committing more inference compute.",
            ]
    (dest / "PHASE3_RESULTS.md").write_text("\n".join(report) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), layout="constrained")
    for ax, key, title in zip(
        axes, ("trace", "retention"), ("Traced response", "Literary retention")
    ):
        for arm in arms:
            for k in (1,) if arm == "dense" else (2, 1):
                series = [v for v in curves if v["arm"] == arm and v["k"] == k]
                ax.plot(
                    [v["targets"] / 1000 for v in series],
                    [v[key]["nll"] for v in series],
                    marker=".",
                    linestyle="--" if arm != "dense" and k == 1 else "-",
                    label=f"{arm} K={k}",
                )
        ax.set(
            xlabel="Supervised tokens (thousands)",
            ylabel="NLL (nats/token)",
            title=title,
        )
    axes[0].legend(fontsize=7)
    fig.savefig(archive / "learning-curves.png", dpi=160)
    plt.close(fig)
    checksum_file = archive / "checksums.json"
    checksums = read(checksum_file)
    checksums["learning-curves.png"] = digest(archive / "learning-curves.png")
    if (archive / "promotion.json").is_file():
        from scripts.phase3_figure import memory_figure, refinement_figure

        figure = refinement_figure(archive)
        checksums[figure.name] = digest(figure)
        memory = memory_figure(archive, chosen)
        checksums[memory.name] = digest(memory)
    save(checksum_file, checksums)
    profiles = [(k, v) for k, v in train.items() if "_profile_" in k]
    batches = [read(p) for p in sorted(archive.glob("batch-*.json"))]
    audit = read(archive / "native-audit.json")
    fusion = read(archive / "export-audit.json")
    perf = [
        "# Phase 3 Metal measurements",
        "",
        "Apple M3 Pro, 14 GPU cores, 18 GiB unified memory. MLX 0.32.4 and MLX-LM 0.32.2. The training path uses the previously validated native GDN backward, BF16 activations and selected-position full-vocabulary cross-entropy. No new kernel was required.",
        "",
        "## Real training updates",
        "",
        "B4, rank 8, layers 10–15, 256-position ceiling. Timing includes evaluated forward, gradients, clipping and AdamW. Data scheduling, validation and checkpoint serialization are outside the update timer. Peak allocation covers training and validation since the last reset.",
        "",
        table(
            [
                "Arm",
                "Updates",
                "Update s",
                "Median [p10,p90] s",
                "Inputs/s",
                "Positions/s",
                "Targets/s",
                "Updates/s",
                "s/M targets",
                "Padding",
                "Peak GiB",
                "Peak RSS GiB",
                "Min available GiB",
                "Swap change GiB",
            ],
            [
                (
                    a,
                    r["step"],
                    f"{r['update_seconds']:.1f}",
                    f"{r['median_s']:.3f} [{r['p10_s']:.3f},{r['p90_s']:.3f}]",
                    f"{r['input_tps']:.0f}",
                    f"{r['positions_s']:.0f}",
                    f"{r['supervised_tps']:.0f}",
                    f"{r['updates_s']:.2f}",
                    f"{r['seconds_per_million']:.0f}",
                    f"{100 * r['padding']:.1f}%",
                    f"{r['peak_bytes'] / 2**30:.2f}",
                    f"{r['rss_peak'] / 2**30:.2f}",
                    f"{r['min_available'] / 2**30:.2f}",
                    f"{(r['swap_end'] - r['swap_start']) / 2**30:+.2f}",
                )
                for a, r in [(a, train["p3_" + a]) for a in arms]
            ],
        ),
        "",
        "## Checkpoint policy screen",
        "",
        "Twelve updates per profile. The initial profiles include startup in the aggregate rate. Ordering changes which examples occur in the first twelve updates. The three protocol-ordered checkpointing profiles share the exact cohort.",
        "",
        table(
            ["Configuration", "Targets", "Targets/s", "Padding", "Peak GiB"],
            [
                (
                    k.removeprefix("p3_direct_profile_"),
                    r["target_tokens"],
                    f"{r['supervised_tps']:.0f}",
                    f"{r['padding']:.1%}",
                    f"{r['peak_bytes'] / 2**30:.2f}",
                )
                for k, r in profiles
            ],
        ),
        "",
        "A second benchmark fixes all 80 real examples and both warmup batches. Each arm performs 20 timed updates after the warmup. Reordering into length groups changes update boundaries. The position cap splits individual microbatches and normalizes by the same four examples’ total contributing tokens.",
        "",
        table(
            [
                "Policy",
                "Microbatch cap",
                "Targets/s",
                "Useful inputs/s",
                "Median update s",
                "Padding",
                "Peak GiB",
            ],
            [
                (
                    r["policy"],
                    r["cap"] or "none",
                    f"{r['targets_s']:.1f}",
                    f"{r['useful_inputs_s']:.1f}",
                    f"{r['median_seconds']:.3f}",
                    f"{r['padding']:.1%}",
                    f"{r['peak'] / 2**30:.2f}",
                )
                for r in batches
            ],
        ),
        "",
        "## Numerical diagnostics and deployment",
        "",
        f"Historical K=1 relative RMS was {audit['historical_k1_relative_rms']:.3g}. Alpha zero was exact in the native check. Gated full versus chunked relative RMS was {audit['cached_full_relative_rms']:.4g} and maximum padded versus single-request RMS was {max(audit['padded_relative_rms']):.4g}. Both optimizer groups restored at step {audit['gate_group_steps'][0]}. Native GDN backward can exhibit atomic-reduction order differences.",
        "",
        f"FP32 adapter fusion into BF16 changed verification-prompt logits by {fusion['verification']['merge_relative_rms']:.2%} relative RMS. The merged model reloaded in an isolated Python process. On 30 traced items, K=2 strings differed on {fusion['results']['different_strings_k2']} items. The unmerged adapter remains the research reference.",
        "",
        "Run-level logs include actual post-update MLX active, cached and peak bytes, process RSS, system availability, swap, gradient norms and group rates. An increase in occupied swap alone cannot establish active thrashing.",
    ]
    perf += [
        "",
        "## Sustained gated run",
        "",
        "The first 325 gated updates used no activation checkpointing. Repeated transient macOS warnings around 2 GiB available prompted a controlled stop at 55,470 targets. The new immutable generation hard-links the same adapter weights and AdamW state. The remaining updates recomputed layer activations. Swap occupancy was approximately 1.86 GiB at the transition. The cohort and optimization schedule are identical across arms. The measured gated cost includes both execution policies.",
        "",
        table(
            [
                "Checkpointing",
                "Updates",
                "Targets",
                "Update seconds",
                "Targets/s",
                "Peak GiB",
            ],
            [
                (
                    key,
                    r["updates"],
                    r["targets"],
                    f"{r['seconds']:.1f}",
                    f"{r['targets'] / r['seconds']:.1f}",
                    f"{r['peak'] / 2**30:.2f}",
                )
                for key, r in train["p3_" + chosen]["checkpoint_segments"].items()
            ],
        ),
    ]
    if (archive / "promotion.json").exists():
        perf += [
            "",
            "## 100K to 250K continuation",
            "",
            "![Per-session MLX peak and system availability during the 250K gated run](measurements/phase3/training-memory.png)",
            "",
            "Each point summarizes 32 consecutive optimizer updates. The band spans post-update system availability. MLX peaks include earlier work in the same worker session. Availability can change while other applications use the Mac.",
            "",
            table(
                [
                    "Arm",
                    "Additional updates",
                    "Additional targets",
                    "Update seconds",
                    "Targets/s",
                ],
                [
                    (
                        arm,
                        train["p3_" + arm + "_250k"]["step"]
                        - train["p3_" + arm]["step"],
                        train["p3_" + arm + "_250k"]["target_tokens"]
                        - train["p3_" + arm]["target_tokens"],
                        f"{train['p3_' + arm + '_250k']['update_seconds'] - train['p3_' + arm]['update_seconds']:.1f}",
                        f"{(train['p3_' + arm + '_250k']['target_tokens'] - train['p3_' + arm]['target_tokens']) / (train['p3_' + arm + '_250k']['update_seconds'] - train['p3_' + arm]['update_seconds']):.1f}",
                    )
                    for arm in ("dense", chosen)
                ],
            ),
        ]
    frontier = json.loads(gzip.decompress((archive / "frontier.json.gz").read_bytes()))
    perf += [
        "",
        "## Traced inference costs",
        "",
        "Sixty traced V2 development requests were split into two counterbalanced halves. Each arm warmed up on a separate short generation. Batch-one request time and batched throughput are reported separately. Times charge every executed lane through the final decode step.",
        "",
        table(
            [
                "Arm / batch",
                "Correct",
                "Total s",
                "Requests/s",
                "Useful output tokens/s",
                "Median batch TTFT s",
                "Median batch duration s",
            ],
            [
                (
                    k,
                    r["correct"],
                    f"{r['seconds']:.2f}",
                    f"{frontier['n'] / r['seconds']:.2f}",
                    f"{sum(v['tokens'] for v in r['records']) / r['seconds']:.1f}",
                    f"{median(v['prefill_seconds'] for v in r['batches']):.4f}",
                    f"{median(v['seconds'] for v in r['batches']):.3f}",
                )
                for k, r in frontier["greedy"].items()
            ],
        ),
        "",
        f"Dense numerical plurality scored {frontier['plurality']['correct']}/{frontier['n']} in {frontier['plurality']['seconds']:.1f} seconds against a gated batch-one budget of {frontier['plurality']['budget']:.1f} seconds. Additional candidates are charged in complete pairs. Selection uses numeric votes with greedy-first ties.",
        "",
    ]
    compact = read(archive / "compaction.json")
    baseline = compact["timing"][0]["records"]
    perf += [
        "## Active-lane filtering",
        "",
        f"On four FP32 continuation steps the largest filtered versus full-batch relative RMS was {max(compact['fp32_filtered_errors']):.3g}. The BF16 timing comparison used 30 identical prompts in counterbalanced order. Primary cohort scoring used fixed lanes.",
        "",
        table(
            [
                "Compaction",
                "Seconds",
                "Lane positions",
                "Answers",
                "Changed strings vs first run",
            ],
            [
                (
                    r["compact"],
                    f"{r['seconds']:.2f}",
                    r["positions"],
                    r["correct"],
                    sum(x["text"] != y["text"] for x, y in zip(baseline, r["records"])),
                )
                for r in compact["timing"]
            ],
        ),
    ]
    projection = read(archive / "projection.json")
    perf += [
        "",
        "## Exact vocabulary-loss timing",
        "",
        f"Four real traces produced input shape {projection['shape']} and {projection['labels']} selected targets. Each method used the same initial model and optimizer, one warmup and four timed updates. Hidden-gradient comparisons isolate projection error from GDN reduction order.",
        "",
        table(
            [
                "Head",
                "Setup s",
                "Warmup s",
                "Median update s",
                "Timed range s",
                "Peak GiB",
                "Loss abs. error",
                "Hidden gradient rel. RMS",
            ],
            [
                (
                    k,
                    f"{r['setup_s']:.3f}",
                    f"{r['warmup_s']:.3f}",
                    f"{r['median_s']:.3f}",
                    f"{min(r['steady_s']):.3f}–{max(r['steady_s']):.3f}",
                    f"{r['peak_bytes'] / 2**30:.2f}",
                    f"{r['abs_mean_loss_error']:.3g}",
                    f"{r['hidden_grad_relative_rms']:.3g}",
                )
                for k, r in projection["results"].items()
            ],
        ),
    ]
    perf += [
        "",
        "## Adapter and merged task performance",
        "",
        table(
            ["Format", "K", "Correct / 30", "Trace NLL", "Language NLL"],
            [
                (
                    m,
                    k,
                    r["generations"]["correct"],
                    f"{r['trace_nll']['nll']:.5f}",
                    f"{r['retention']['nll']:.5f}",
                )
                for m in ("adapter", "merged")
                for k, r in fusion["results"][m].items()
            ],
        ),
    ]
    if (archive / "export-audit-250k.json").is_file():
        promoted_fusion = read(archive / "export-audit-250k.json")
        results = promoted_fusion["results"]
        perf += [
            "",
            "## Promoted 250K deployment",
            "",
            f"FP32 fusion followed by BF16 storage yielded verification-prompt relative RMS {promoted_fusion['verification']['merge_relative_rms']:.2%}. K=2 generation changed on {results['different_strings_k2']}/30 prompts and K=1 on {results['different_strings_k1']}/30.",
            "",
            table(
                ["Format", "Depth", "Correct / 30", "Trace NLL", "Language NLL"],
                [
                    (
                        mode,
                        k,
                        r["generations"]["correct"],
                        f"{r['trace_nll']['nll']:.5f}",
                        f"{r['retention']['nll']:.5f}",
                    )
                    for mode in ("adapter", "merged")
                    for k, r in results[mode].items()
                ],
            ),
        ]
    (dest / "PHASE3_OPTIMIZATIONS.md").write_text("\n".join(perf) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path)
    p.add_argument("--destination", type=Path)
    a = p.parse_args()
    archive = a.archive or Path(read("outputs/phase3/archive-location.json")["archive"])
    dest = a.destination or (
        archive.parents[1]
        if archive.parent.name == "measurements"
        else archive / "reports"
    )
    docs(archive, dest)
