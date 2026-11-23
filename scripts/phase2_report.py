"""Render the final research tables from measured, completed runs."""

from pathlib import Path
from statistics import mean

from loopqwen.checkpoint import location
from loopqwen.io import read

ROOT = Path("outputs/phase2")


def scores(group, arm):
    return read(ROOT / group / (arm + ".json"))


def state(arm):
    return read(location(Path("checkpoints") / arm) / "state.json")


def create():
    m = read("experiments/phase2_confirm.json")
    sel = read("experiments/phase2_selection.json")
    paired = read(ROOT / "phase2_confirm/paired.json")
    public = read(ROOT / "transfer.json")
    cost = read(ROOT / "compute.json")
    rows = []

    def p(s=""):
        rows.append(s + "\n")

    def table(columns, values):
        p("| " + " | ".join(columns) + " |")
        p("|" + "|".join(["---"] * len(columns)) + "|")
        for row in values:
            p("| " + " | ".join(map(str, row)) + " |")
        p()

    p("# Phase 2 results")
    p()
    p(
        f"The development criterion selected **{sel['selected']}**. The confirmatory runs use seeds 43, 53 and 67 with 84 data-matched updates per model. Each time-matched dense run continued until it reached its paired loop’s measured update time. The research question is whether shared recurrent depth improves exact answers, likelihood or quality for the compute spent after matched adaptation."
    )
    p()
    p("## Setup")
    p()
    p(
        "Apple M3 Pro, 14 GPU cores, 18 GiB unified memory. MLX 0.32.4 and MLX-LM 0.32.2. All forwards, gradients and optimizer updates used Metal. The official Qwen3.5-0.8B-Base revision was `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. Its text model has 752,393,024 original parameters. Every additional layer execution uses the original module and per-site cache. [Architecture and data audit](PHASE2_ARCHITECTURE.md)"
    )
    p()
    p(
        "The corpus contains 24,000 training problems, 450 development problems and two disjoint 300-problem ID/OOD splits. Training on every example once would supply 100,453 answer-only labels or 1,784,049 trace labels. The confirmatory protocol scores the first 150 stratified problems per split for NLL, 90 for direct answers and 45 for traced answers. The trace cap is 128 output tokens. Prose NLL uses the same 48 independent paragraphs for all confirmatory arms. [Data manifest](measurements/phase2/data-manifest.json)"
    )
    p()
    p("## Development and design selection")
    p()
    p(
        "The historical 200-update dense/loop result was 61/50 correct and the 1,000-update result was 65/59 correct, each on 110 older questions. V2 has different tasks, prompts and golds. The first V2 experiment exposed each model to the same 288 problems. Answer-only supervision scored 1,220 targets and trace supervision scored 21,834. [Historical results](RESULTS.md) · [V2 Stage A table](measurements/phase2/phase2_direct/comparison.md)"
    )
    p()
    p(
        "The short screens held the update count at 24. Placement used only compatible MLP projections to keep parameter counts identical across different hybrid spans. Model selection used development answer-only NLL. Each candidate competed on the same examples. [Complete screen](measurements/phase2/phase2_screen/comparison.md) · [Frozen policy](../experiments/phase2_selection.json)"
    )
    p()
    p(
        "![Development likelihood by training policy](measurements/phase2/phase2_screen/loss.png)"
    )
    p()
    choices = read("experiments/phase2_screen.json")["runs"]
    table(
        [
            "Policy",
            "Trainable",
            "K=2 updates",
            "Alpha at last update",
            "Maximum preclip norm",
            "Update seconds",
        ],
        (
            [
                a.removeprefix("p2_screen_"),
                s["trainable"],
                sum(h["depths"] == [2] for h in s["history"]),
                f"{s['history'][-1]['alpha']:.5f}"
                if s["history"][-1]["alpha"] is not None
                else "—",
                f"{max(h['gradient_norm'] for h in s['history']):.2f}",
                f"{sum(h['seconds'] for h in s['history']):.1f}",
            ]
            for a in choices
            for s in [state(a)]
        ),
    )
    p("## Frozen ID and OOD comparisons")
    p()
    table(
        [
            "Split",
            "Prompt",
            "Control",
            "Dense accuracy mean ± SD",
            "Loop accuracy mean ± SD",
            "Loop − dense, paired 95% CI (pp)",
        ],
        (
            [
                *key.split("/"),
                f"{value['accuracy']['dense']['mean']:.1%} ± {value['accuracy']['dense']['sd']:.1%}",
                f"{value['accuracy']['loop']['mean']:.1%} ± {value['accuracy']['loop']['sd']:.1%}",
                f"{100 * value['mean_difference']:+.1f} [{100 * value['paired_ci95'][0]:+.1f}, {100 * value['paired_ci95'][1]:+.1f}]",
            ]
            for key, value in paired.items()
        ),
    )
    p(
        "SD is across three training runs. Intervals jointly resample seeds and category-stratified items, with the same item resample for every seed and model. Three seeds and shared procedural templates limit precision. Per-seed wins/losses, exact McNemar probabilities, difficulty and reasoning-depth breakdowns accompany the per-example completions. [Paired statistics](measurements/phase2/phase2_confirm/paired.json) · [Per-model scores](measurements/phase2/phase2_confirm/comparison.md)"
    )
    p()
    p("## Compute and retention")
    p()
    base = public["base_prose_test"]["nll"]
    p(
        f"The original model's general-language NLL on the final prose slice was {base:.4f}. The projection-work estimate counts two FLOPs per multiply-add for executed backbone and adapter linear projections and the selected-token vocabulary head. GDN state updates, attention products, backward, optimizer and checkpoint recomputation contribute additional work. Time matching uses synchronized full training updates including those operations. Driver utilization samples describe the whole GPU."
    )
    p()
    vals = []
    for a in m["runs"]:
        r = scores("phase2_confirm", a)
        t = r["training"]
        v = r["scores"]["test"]["language"]["nll"]
        f = cost[a]
        vals.append(
            [
                a.removeprefix("p2_confirm_"),
                t["step"],
                t["target_tokens"],
                t["input_tokens"],
                f"{t['update_seconds']:.1f}",
                f"{t['input_tokens'] / t['update_seconds']:.0f}",
                f"{t['peak_bytes'] / 2**30:.2f}",
                f"{f['forward_projection_flops'] / 1e12:.1f}",
                f"{v:.3f} ({v - base:+.3f})",
            ]
        )
    table(
        [
            "Seed / arm",
            "Updates",
            "Targets",
            "Inputs",
            "Update s",
            "Input tok/s",
            "MLX GiB",
            "Forward projection TFLOPs",
            "Prose NLL (Δ base)",
        ],
        vals,
    )
    p(
        "For the sequential inference control, the dense model first answers each of 30 fixed cases greedily. It spends remaining elapsed time on two temperature-0.7 candidates per case in predetermined order. Plurality prefers the greedy answer in a tie. The last pair can overshoot the budget. A dense run whose greedy time already exceeds the loop budget receives zero extra candidates."
    )
    p()
    table(
        [
            "Seed",
            "Loop s",
            "Dense greedy s",
            "Dense vote s",
            "Extra pairs",
            "Correct loop/greedy/vote",
        ],
        (
            [
                r["seed"],
                f"{r['loop_seconds']:.2f}",
                f"{r['dense_seconds']:.2f}",
                f"{r['strategy_seconds']:.2f}",
                r["candidate_sets"],
                f"{r['loop']}/{r['dense']}/{r['plurality']}",
            ]
            for r in read(ROOT / "frontier.json")
        ),
    )
    p("## External transfer and depth")
    p()
    table(
        ["Seed 43 arm", "Dataset", "Correct / n", "Answer NLL"],
        (
            [a, k, f"{g['correct']}/{g['n']}", f"{s['loss']['nll']:.3f}"]
            for a, tests in public["runs"].items()
            for k, s in tests.items()
            for g in [s["generations"]]
        ),
    )
    p(
        "External scores use raw answer prompts and bounded greedy decoding. MATH uses a fixed scalar-answer, non-diagram subset and GSM8K uses the publisher's test split. These are exploratory transfer measurements."
    )
    p()
    table(
        [
            "K",
            "Effective layers",
            "Development correct / 30",
            "Answer NLL",
            "Decode tok/s after 512-position prefill",
        ],
        (
            [
                k,
                r["executions"],
                r["greedy"]["correct"],
                f"{r['development_loss']['nll']:.3f}",
                f"{next(t for t in r['timing'] if t['length'] == 512)['median_decode_tps']:.1f}",
            ]
            for k, r in read(ROOT / "depth_latency.json").items()
        ),
    )
    p(
        "K=3 is a depth extrapolation. All three depths above use the same seed-43 checkpoint."
    )
    p()
    p("## Numerical and performance diagnostics")
    p()
    profile = read(ROOT / "profile.json")["runs"]
    table(
        [
            "Training positions",
            "Batch",
            "Checkpoint",
            "Median s/update",
            "MAD s",
            "Input tok/s",
            "Peak MLX GiB",
            "RSS GiB",
            "Available system GiB",
        ],
        (
            [
                r["length"],
                r["batch"],
                r["checkpointing"],
                f"{r['median']:.3f}",
                f"{r['mad']:.3f}",
                f"{r['input_tps']:.0f}",
                f"{r['peak'] / 2**30:.2f}",
                f"{r['rss'] / 2**30:.2f}",
                f"{r['available'] / 2**30:.2f}",
            ]
            for r in profile
        ),
    )
    rep = read(ROOT / "repeat.json")
    p(
        f"The repeated 24-update native run used the same seed and ordering. Its trained LoRA B factors differed by {rep['rank_factor_b_relative']:.2%} relative RMS. The two trace NLLs were {rep['validation_nll'][0]:.4f} and {rep['validation_nll'][1]:.4f}. Correct answer counts were {rep['answer_accuracy'][0]} and {rep['answer_accuracy'][1]} of {rep['n']}. [Native precision and cache audit](PHASE2_OPTIMIZATIONS.md)"
    )
    p()
    base_dev = read(ROOT / "base_dev.json")
    table(
        [
            "Unadapted span / depth",
            "Development answer NLL",
            "Prose NLL",
            "Mean last-pass cosine",
            "Mean relative hidden change",
        ],
        (
            [
                k,
                f"{v['metrics']['answer']['loss']['nll']:.3f}",
                f"{v['metrics']['language']['nll']:.3f}",
                f"{mean(seq[-1]['cosine'] for seq in v['representations']):.3f}",
                f"{mean(seq[-1]['relative_change'] for seq in v['representations']):.3f}",
            ]
            for k, v in base_dev.items()
        ),
    )
    anchor = read(ROOT / "anchor_diagnostics.json")
    p(
        f"The separate input-anchor ablation trained a scalar mixing the original block input into the next block application. Alpha moved from {anchor['initial_alpha']:.3f} to {anchor['final_alpha']:.5f}. K=1 parity was exact and K=2 cached versus uncached logit RMS was {anchor['cache_relative_rms']:.2%}. This arm was outside confirmatory selection. [Matched development comparison](measurements/phase2/phase2_anchor/comparison.md)"
    )
    p()
    p("## Export parity")
    p()
    table(
        [
            "Seed 43 form",
            "Answer NLL",
            "Correct / 30",
            "Prose NLL",
            "Token top-1 divergence",
            "Greedy text divergence",
        ],
        (
            [
                k,
                f"{r['answer_nll']['nll']:.4f}",
                r["greedy"]["correct"],
                f"{r['language']['nll']:.4f}",
                f"{r.get('top1_disagreement', 0):.1%}",
                f"{r.get('generation_disagreement', 0):.1%}",
            ]
            for k, r in read(ROOT / "merge_audit.json").items()
        ),
    )
    p(
        "Both BF16 fusions include the recurrent loader, tokenizer and provenance. Clean subprocesses checked their merged logits. Adapter inference preserves the original unfused computation. The seed-43 data-matched pair is stored under `adapters/p2_confirm_43_{dense,loop}`. Timestamped local training generations under `checkpoints/` include AdamW state."
    )
    p()
    p("## Reproduction")
    p()
    p("Run from a fresh checkout on an arm64 Mac with MLX Metal.")
    p()
    p(
        "```sh\nuv sync --locked --group figures\n.venv/bin/python -m loopqwen download\nscripts/run_phase2.sh\n.venv/bin/python -m loopqwen compare --experiment phase2_confirm\n.venv/bin/python -m loopqwen generate --checkpoint adapters/p2_confirm_43_loop --prompt 'Evaluate ((8 + 4) * 3 - 5).\nAnswer:' --max-tokens 24\n```"
    )
    Path("docs/PHASE2_RESULTS.md").write_text("\n".join(rows))


if __name__ == "__main__":
    create()
