"""Summarize every sealed on-policy sample, real execution and leave-one-out update."""

from collections import Counter
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typst
from pypdf import PdfReader

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.table_policy_learning import NAME
from loopqwen.research.table_worlds import SOURCE
from scripts.assistance_deliberation import write_csv

P = Path("paper")
R = Path("research/results") / NAME


def measure():
    s = read(R / "registration.json")
    decision = read(R / "decision.json")
    index = read(R / "checksums.json")
    interrupted = decision.get("status") == "interrupted"
    filename = "partial-state.json.gz" if interrupted else "on-policy-rollouts.json.gz"
    assert digest(R / filename) == index[filename]
    payload = unpack(R / filename)
    log = payload["rollouts"] if interrupted else payload
    assert log
    tok = tokenizer(s)
    stops = set(tok.eos_token_ids) | {tok.convert_tokens_to_ids("<|im_end|>")}
    questions = {r["id"]: r for r in read(SOURCE)["cohorts"]["train"]}
    draws, steps, pairs = [], [], []
    totals = Counter()
    examples = {}
    for group in log:
        step = group["step"]
        w = questions[group["question"]]
        phase = group["stage"]
        sql = phase == "query"
        totals[phase + "_updates"] += 1
        for j, (prefix, batch) in enumerate(
            zip(group["prefixes"], group["draws"], strict=True)
        ):
            for i, draw in enumerate(batch):
                tokens = draw["tokens"]
                eos = tokens[-1] in stops
                decoded = tok.decode(tokens[:-1] if eos else tokens)
                answer = decoded.partition("</think>")[2].strip() if eos else ""
                details = group["details"][i] if sql else group["details"][j][i]
                results = details["results"] if sql else []
                right = sum(o["correct"] for o in results)
                used_both = len(results) == 2 and all(o["used"] for o in results)
                print_name = "query" if sql else ("original" if j == 0 else "altered")
                data = {
                    "update": step,
                    "question": w["id"],
                    "kind": print_name,
                    "draw": i,
                    "input_tokens": len(prefix),
                    "emitted_tokens": len(tokens),
                    "natural_stop": eos,
                    "thinking_closed": eos and "</think>" in decoded,
                    "extra_closing_markers": max(0, decoded.count("</think>") - 1),
                    "visible": answer,
                    "statement": details["statement"] if sql else None,
                    "read_both": used_both if sql else None,
                    "correct_sql_worlds": right if sql else None,
                    "correct_answer": details.get("correct"),
                    "opposite": details.get("opposite"),
                    "local_reward": details["reward"],
                    "joint_reward": group["rewards"][i],
                    "advantage": group["advantages"][i],
                }
                draws.append(data)
                if sql:
                    totals["valid_sql"] += used_both
                    totals["any_gold_sql"] += bool(right)
                    totals["both_sql"] += right == 2
                    if right and right not in examples:
                        examples[right] = {
                            "question": w["question"],
                            "source": w["id"],
                            "sql": details["statement"],
                            "gold": [v["answer"] for v in w["versions"]],
                            "obtained": [
                                str(x["observation"]["rows"])[:220]
                                if "observation" in x
                                else x["error"]
                                for x in results
                            ],
                        }
                else:
                    assert details["visible"] == answer
                    totals[print_name + "_correct"] += details["correct"]
                    totals[print_name + "_opposite"] += details["opposite"]
        if not sql:
            for i, (a, b) in enumerate(zip(*group["details"], strict=True)):
                both = a["correct"] and b["correct"]
                same = bool(a["visible"]) and a["visible"] == b["visible"]
                totals["both_answers"] += both
                totals["identical_nonempty"] += same
                pairs.append(
                    {
                        "update": step,
                        "question": w["id"],
                        "draw": i,
                        "correct_original": a["correct"],
                        "correct_altered": b["correct"],
                        "both_correct": both,
                        "identical_nonempty": same,
                        "joint_reward": group["rewards"][i],
                    }
                )
        varied = len(set(group["rewards"])) > 1
        steps.append(
            {
                "update": step,
                "question": w["id"],
                "stage": phase,
                "mean_return": mean(group["rewards"]),
                "varied": varied,
                "policy_gradient_norm": group["policy_gradient_norm"],
                "total_gradient_norm": group["gradient_norm"],
                "anchor_nll": group["anchor_nll"],
                "logp_error": group["sampled_logp_mean_abs_replay_error"],
                "sampled_tokens": sum(
                    len(r["tokens"]) for batch in group["draws"] for r in batch
                ),
            }
        )
    write_csv(P / "table-policy-learning-draws.csv", draws)
    write_csv(P / "table-policy-learning-steps.csv", steps)
    if pairs:
        write_csv(P / "table-policy-learning-paired-answers.csv", pairs)
    n = sum(s["group_size"] * len(g["draws"]) for g in log)
    assert n == len(draws)
    measured = decision.get("measured")
    if measured:
        assert measured["sampled_tokens"] == sum(d["emitted_tokens"] for d in draws)
        assert measured["nonzero_policy_gradient"] == sum(
            g["policy_gradient_norm"] > 0 for g in steps
        )
        assert measured["unequal_returns"] == sum(g["varied"] for g in steps)
    verification = read("docs/measurements/table-policy-native-sampling.json")
    assert verification["sealed_rollouts"] == index[filename]
    jobs = [
        j
        for f in Path("research/sessions").glob("window-*.json")
        for j in read(f)["jobs"]
        if j["job"] == NAME
    ]
    assert jobs
    to_display = []
    for kind in ("query", "original", "altered"):
        d = [r for r in draws if r["kind"] == kind]
        if d:
            to_display.append(
                [
                    kind.title(),
                    len(d),
                    sum(r["emitted_tokens"] for r in d),
                    sum(r["natural_stop"] for r in d),
                    sum(r["thinking_closed"] for r in d),
                    sum(bool(r["visible"]) for r in d),
                ]
            )
    logged_drift = [r["logp_error"] for r in steps if r["logp_error"] is not None]
    result = {
        "passed": decision["passed"],
        "interrupted": interrupted,
        "steps": len(log),
        "draws": n,
        "tokens": sum(r["emitted_tokens"] for r in draws),
        "varied": sum(g["varied"] for g in steps),
        "positive_pg": sum(r["policy_gradient_norm"] > 0 for r in steps),
        "correct_groups": sum(
            any(any(o["correct"] for o in d["results"]) for d in g["details"])
            if g["stage"] == "query"
            else any(x["correct"] for b in g["details"] for x in b)
            for g in log
        ),
        "drift": f"{median(logged_drift):.5f}" if logged_drift else "unavailable",
        "pair_count": len(pairs),
        "query_count": sum(r["kind"] == "query" for r in draws),
        "categories": to_display,
        "counts": {
            k: totals[k]
            for k in (
                "query_updates",
                "answer_updates",
                "valid_sql",
                "any_gold_sql",
                "both_sql",
                "original_correct",
                "altered_correct",
                "original_opposite",
                "altered_opposite",
                "both_answers",
                "identical_nonempty",
            )
        },
        "seconds": round(sum(j["seconds"] for j in jobs), 1),
        "peak_gib": round(max(j["peak_bytes"] for j in jobs) / 2**30, 3),
        "reason": decision.get("reason"),
        "conditions": [
            [k, "Pass" if v else "Miss"] for k, v in decision.get("gates", {}).items()
        ],
        "examples": [
            {**v, "correct_worlds": k}
            for k, v in sorted(examples.items(), reverse=True)
        ],
        "repeat_tokens": verification["sampled_tokens"],
        "repeat_matched": verification["matching_tokens"],
        "max_err": f"{verification['max_abs_logp_error']:.8f}",
        "archive": digest(R / "checksums.json"),
    }
    save(P / "table-policy-learning-paper.json", result)
    return steps, result


def build():
    steps, summary = measure()
    plt.rcParams.update(
        {
            "svg.hashsalt": "native-table-policy-gradients",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), layout="constrained")
    for stage, color in (("answer", "#11705e"), ("query", "#9b532d")):
        rows = [r for r in steps if r["stage"] == stage]
        bins = [
            [r for r in rows if i <= r["update"] < i + 16]
            for i in range(1, len(steps) + 1, 16)
        ]
        included = [batch for batch in bins if batch]
        if included:
            x = [mean(r["update"] for r in b) for b in included]
            axes[0].plot(
                x,
                [mean(r["mean_return"] for r in b) for b in included],
                color=color,
                marker="o",
                ms=2,
                label=stage.title(),
            )
            axes[1].plot(
                x,
                [mean(r["policy_gradient_norm"] > 0 for r in b) for b in included],
                color=color,
                marker="o",
                ms=2,
            )
    axes[0].set(
        xlabel="Native optimizer update", ylabel="Mean return of four paired draws"
    )
    axes[0].legend(frameon=False)
    axes[1].set(
        xlabel="Native optimizer update",
        ylabel="Groups with execution gradient",
        ylim=(-0.05, 1.05),
    )
    fig.savefig(P / "table-policy-learning.svg", metadata={"Date": None})
    plt.close(fig)
    path = P / "table-policy-learning.pdf"
    pdf = typst.compile(str(P / "table-policy-learning.typ"), timestamp=0)
    assert pdf == typst.compile(str(P / "table-policy-learning.typ"), timestamp=0)
    path.write_bytes(pdf)
    pages = PdfReader(path).pages
    assert all(len(p.extract_text()) > 280 for p in pages)
    print(len(pages), "pages", summary["draws"], "actual trajectories", digest(path))


if __name__ == "__main__":
    build()
