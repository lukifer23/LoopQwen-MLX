"""Report the transfer and retention strata of the single resident folded decoder."""

import csv

import matplotlib.pyplot as plt

from loopqwen.evaluation import executable_tools, table_questions
from loopqwen.evaluation.mechanism import unpack
from loopqwen.tool_session import compact
from loopqwen.io import digest, read


def user_evidence(item, trajectory):
    if item["family"] == "database":
        [reference] = table_questions.reference(item)
    elif item["family"] == "public":
        _, reference = executable_tools.oracle(item)
    else:
        reference = item["gold"]
    history = [
        {k: t[k] for k in ("call", "observation") if k in t}
        for t in trajectory["turns"]
        if "call" in t
    ]
    return [item.get("user_request", item["question"]), reference, compact(history)]


def make(paper, results, result, finish, name="folded-system-01"):
    unified = name == "shared-session-tokenizer-replay-01"
    shared = name.startswith("shared-session-")
    prefix = (
        "unified-session"
        if unified
        else "shared-session"
        if shared
        else "folded-system"
    )
    root = results / name
    if not (root / "checksums.json").is_file():
        return
    d = result(name)
    all_groups = d["groups"]
    counts = list(all_groups["all"]["candidate"])
    labels = {
        "fresh": "Fresh human questions",
        "math": "Word problems",
        "judgment": "BoolQ",
        "public": "Numerical functions",
        "database": "SQLite",
        "file": "Project files",
        "package": "Installed packages",
        "all": "All requests",
    }
    if shared:
        labels = {
            **labels,
            "arithmetic": "Field calculator",
            "reading": "Field reading",
        }
    if unified:
        labels["user_extraction"] = "Human extraction"
    with (paper / (prefix + ".csv")).open("w") as output:
        w = csv.writer(output)
        w.writerow(
            [
                "stratum",
                "side",
                *counts,
                "paired_gains",
                "paired_losses",
                "mcnemar_p",
                "cluster_p",
            ]
        )
        for key in labels:
            group = all_groups[key]
            for side in ("candidate", "control"):
                w.writerow(
                    [
                        key,
                        side,
                        *[group[side][k] for k in counts],
                        group["gains"],
                        group["losses"],
                        group["p"],
                        group["cluster_p"],
                    ]
                )
    checks = read(root / "checksums.json")
    subjects = read(read(root / "registration.json")["material"])["rows"]
    request_file = (prefix + "-requests.csv") if shared else "folded-requests.csv"
    with (paper / request_file).open("w") as output:
        w = csv.writer(output)
        w.writerow(
            [
                "id",
                "family",
                "source",
                "side",
                "user_success",
                "correct_final",
                "completed",
                "grounded",
                "seconds",
                "first_token_s",
                "peak_bytes",
                "return_status",
                "answer",
            ]
            + (["route", "external_probability"] if shared else [])
            + ["request", "reference", "calls_and_observations"]
        )
        for side in ("candidate", "control"):
            f = root / (side + ".json.gz")
            assert digest(f) == checks[f.name]
            for rec, item in zip(unpack(f), subjects, strict=True):
                assert rec["id"] == item["id"]
                scores = rec["score"]
                w.writerow(
                    [
                        rec["id"],
                        item["family"],
                        item["source"],
                        side,
                        *[
                            scores[k]
                            for k in ("success", "answer", "completed", "grounded")
                        ],
                        rec["wall_s"],
                        rec["first_token_s"],
                        rec["peak"],
                        rec["trajectory"]["status"],
                        rec["trajectory"]["answer"],
                    ]
                    + (
                        [
                            rec["trajectory"].get("short_policy_route"),
                            rec["trajectory"].get("external_probability"),
                        ]
                        if shared
                        else []
                    )
                    + user_evidence(item, rec["trajectory"])
                )
    fig, (score, latency) = plt.subplots(1, 2, figsize=(7.3, 3.3), sharey=True)
    for i, (key, label) in enumerate(labels.items()):
        g = all_groups[key]
        for side, direction, color, marker, title in (
            (
                "candidate",
                -0.10,
                "#11675d",
                "o",
                "Unified decoder"
                if unified
                else "Shared views"
                if shared
                else "Folded",
            ),
            ("control", 0.10, "#9d5932", "s", "Untouched Base"),
        ):
            v = g[side]
            opts = {
                "marker": marker,
                "color": color,
                "s": 28,
                "label": title if i == 0 else None,
            }
            score.scatter(v["success"] / v["n"] * 100, i + direction, **opts)
            latency.scatter(v["seconds"] / v["n"], i + direction, **opts)
    score.set_yticks(range(len(labels)), labels.values())
    score.invert_yaxis()
    score.set_xlabel("Correct user outcomes (%)")
    score.set_xlim(-4, 104)
    latency.set_xlabel("Measured seconds per request")
    latency.set_xlim(left=0)
    score.legend(
        frameon=False, loc="lower left", bbox_to_anchor=(-0.07, 1), ncol=2, fontsize=8
    )
    finish(fig, prefix)


def shared_objective(paper, results, result):
    """Replay optimizer draws and sum the per-task cross-entropy weights actually used."""
    import random
    from collections import defaultdict
    from mlx_lm.utils import load_tokenizer
    from loopqwen.model_loader import local_snapshot

    name = "shared-session-01"
    root = results / name
    if not (root / "checksums.json").exists():
        return
    result(name)
    spec = read(root / "registration.json")
    data = read(spec["training"])["examples"]
    tok = load_tokenizer(local_snapshot())
    lengths = []
    stats = defaultdict(lambda: [0, 0, 0, 0, 0.0])
    for w in data:
        text = (
            (w["thinking"] + "\n" if w["thinking"] else "")
            + "</think>\n\n"
            + w["visible"]
        )
        n = len(tok.encode(text, add_special_tokens=False)) + 1
        lengths.append(n)
        s = stats[w["source"]]
        s[0] += 1
        s[1] += n
    rng = random.Random(spec["fit_seed"])
    for _ in range(spec["updates"]):
        ix = rng.sample(range(len(data)), spec["batch"])
        total = sum(lengths[i] for i in ix)
        for i in ix:
            s = stats[data[i]["source"]]
            s[2] += 1
            s[3] += lengths[i]
            s[4] += lengths[i] / total / spec["updates"]
    assert sum(s[3] for s in stats.values()) == read(root / "fit.json")["targets"]
    assert abs(sum(s[4] for s in stats.values()) - 1) < 1e-8
    with (paper / "shared-objective-weights.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "source",
                "fitting_examples",
                "available_response_tokens",
                "sampled_examples",
                "sampled_response_tokens",
                "fraction_of_averaged_loss_weight",
            ]
        )
        for source, values in sorted(stats.items()):
            w.writerow([source, *values])


def workload_mix(paper, results, result, finish):
    """Standardize measured fresh and retention outcomes to different request mixes."""
    names = ("shared-session-01", "shared-session-tokenizer-replay-01")
    names = [n for n in names if (results / n / "checksums.json").exists()]
    if not names:
        return
    fig, axes = plt.subplots(
        1, len(names), figsize=(3.65 * len(names), 2.6), squeeze=False
    )
    with (paper / "request-mix.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "study",
                "side",
                "fresh_correct",
                "fresh_n",
                "retention_correct",
                "retention_n",
                "measured_fresh_fraction",
                "measured_accuracy",
                "fresh_fraction_at_equal_accuracy",
            ]
        )
        for name, ax in zip(names, axes[0], strict=True):
            groups = result(name)["groups"]
            rates = {}
            observed = {}
            for side in ("candidate", "control"):
                total = groups["all"][side]
                fresh = groups["fresh"][side]
                kept = total["success"] - fresh["success"]
                n_keep = total["n"] - fresh["n"]
                rates[side] = (kept / n_keep, fresh["success"] / fresh["n"])
                observed[side] = (fresh["success"], fresh["n"], kept, n_keep)
            a, b = rates.values()
            delta_kept, delta_fresh = a[0] - b[0], a[1] - b[1]
            cross = None
            if delta_kept != delta_fresh:
                x = delta_kept / (delta_kept - delta_fresh)
                if 0 <= x <= 1:
                    cross = x
            for side, color, title in (
                ("candidate", "#11675d", "Recurrent"),
                ("control", "#9d5932", "Base"),
            ):
                yes, n_new, kept, n_kept = observed[side]
                n = n_new + n_kept
                share = n_new / n
                score = (yes + kept) / n
                writer.writerow(
                    [name, side, yes, n_new, kept, n_kept, share, score, cross]
                )
                ax.plot(
                    [0, 100], [100 * v for v in rates[side]], color=color, label=title
                )
                ax.scatter(100 * share, 100 * score, color=color, s=24, zorder=3)
            if cross is not None:
                ax.axvline(100 * cross, ls=":", color="#686868", lw=1)
            ax.set(
                xlim=(0, 100),
                ylim=(0, 100),
                xlabel="Fresh human requests (%)",
                title="Two shared views"
                if name == "shared-session-01"
                else "One unified decoder",
            )
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Standardized correct outcomes (%)")
    axes[0, 0].legend(loc="best", frameon=False, fontsize=8)
    finish(fig, "request-mix")


def repeated_requests(paper, results, result):
    """Compare the two service designs only on requests that actually recur."""
    from scripts.field_cli_replay import identical

    studies = ("shared-session-01", "shared-session-tokenizer-replay-01")
    if not all((results / n / "checksums.json").exists() for n in studies):
        return
    panel, arms = [], []
    for study in studies:
        result(study)
        root = results / study
        protocol = read(root / "registration.json")
        panel.append({w["id"]: w for w in read(protocol["material"])["rows"]})
        inventory = read(root / "checksums.json")
        pair = {}
        for side in ("candidate", "control"):
            trace = root / (side + ".json.gz")
            assert digest(trace) == inventory[trace.name]
            pair[side] = {r["id"]: r for r in unpack(trace)}
            assert pair[side].keys() == panel[-1].keys()
        arms.append(pair)
    ids = sorted(panel[0].keys() & panel[1].keys())
    assert len(ids) == 72
    with (paper / "repeated-service-requests.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "id",
                "family",
                "input_sha256",
                "earlier_success",
                "joint_success",
                "earlier_base_success",
                "joint_base_success",
                "candidate_trace_identical",
                "base_trace_identical",
            ]
        )
        for key in ids:
            old, new = [p["candidate"][key] for p in arms]
            old_base, new_base = [p["control"][key] for p in arms]
            assert panel[0][key]["initial_ids"] == panel[1][key]["initial_ids"]
            assert len({r["input_sha256"] for r in (old, new, old_base, new_base)}) == 1
            writer.writerow(
                [
                    key,
                    panel[0][key]["family"],
                    old["input_sha256"],
                    *[r["score"]["success"] for r in (old, new, old_base, new_base)],
                    identical(old["trajectory"], new["trajectory"]),
                    identical(old_base["trajectory"], new_base["trajectory"]),
                ]
            )
