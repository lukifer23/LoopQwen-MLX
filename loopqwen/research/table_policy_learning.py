"""Update the recurrent policy from its own MLX samples and independent table denotations."""

import math
import shutil
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from .. import checkpoint, qwen_tools
from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import digest, read, save
from ..kernels.gdn_backend import select
from ..reentry import from_base
from ..relational_tools import tools
from ..tool_session import compact
from .core import Budget
from .cross_view_study import tokenizer
from .file_evidence_environment import capture
from .integrated_assistance_fit import positions
from .registered import registered
from .table_reentry_fit import (
    CODES,
    accumulate,
    apply,
    decks,
    gold_nll,
    inputs,
    optimizer,
    parity,
    root_for,
)
from .table_worlds import (
    FILENAME,
    SOURCE,
    locations,
    prepare_tables,
    present,
    reward,
    state,
)

NAME = "table-execution-learning-01"
PARENT = "table-reentry-fit-01"


def advantages(rewards):
    if len(rewards) < 2:
        raise ValueError("Leave-one-out requires multiple independent draws")
    total = sum(rewards)
    return [r - (total - r) / (len(rewards) - 1) for r in rewards]


def sample(net, prefix, stop, count, cap, temperature, seed, guard):
    # Equal-length inputs avoid padding and ensure independent recurrent histories per trajectory.
    net.eval()
    cache = net.make_cache()
    ids = mx.broadcast_to(mx.array([prefix]), (count, len(prefix)))
    key = mx.random.key(seed)
    rows = [{"tokens": [], "behavior_logp": []} for _ in range(count)]
    active = [True] * count
    for step in range(cap):
        if step % 16 == 0:
            guard.check()
        x = net.hidden(ids, cache)[:, -1, :]
        logits = net.project(x).astype(mx.float32) / temperature
        key, draw = mx.random.split(key)
        ids = mx.random.categorical(logits, key=draw)
        chosen = mx.take_along_axis(logits, ids[:, None], axis=-1)[:, 0] - mx.logsumexp(
            logits, axis=-1
        )
        mx.eval(ids, chosen)
        for j, (token, lp) in enumerate(
            zip(ids.tolist(), chosen.tolist(), strict=True)
        ):
            if active[j]:
                rows[j]["tokens"].append(token)
                rows[j]["behavior_logp"].append(lp)
                active[j] = token not in stop
        if not any(active):
            break
        ids = ids[:, None]
    return rows


def visible(tok, row, stop):
    y = row["tokens"]
    if y[-1] not in stop:
        return "", False
    _, sep, text = tok.decode(y[:-1]).partition("</think>")
    return text.strip(), bool(sep)


def contextual(w, folder):
    messages, defs = state(w, folder)
    call = {
        "name": "database_query",
        "arguments": {
            "database_name": FILENAME,
            "statement": "SELECT * FROM records ORDER BY rowid",
        },
    }
    pairs = []
    for path in locations(folder, w):
        _, query = tools([path])
        observation = query.callback(**call["arguments"])
        assert observation["rows"] and not observation["truncated"]
        pairs.append(
            [
                *messages,
                qwen_tools.assistant_message([call]),
                {"role": "tool", "content": compact(observation)},
            ]
        )
    return messages, defs, pairs


def scored(tok, w, tables, stage, rows, stop, defs):
    if stage == "query":
        details = []
        for row in rows[0]:
            text, closed = visible(tok, row, stop)
            details.append(reward(text, closed, defs, w, tables))
        return [x["reward"] for x in details], details
    results = []
    for world, batch in enumerate(rows):
        observations = []
        for draw in batch:
            text, closed = visible(tok, draw, stop)
            right = closed and present(text, w["versions"][world]["answer"])
            opposite = closed and present(text, w["versions"][1 - world]["answer"])
            observations.append(
                {
                    "visible": text,
                    "closed": closed,
                    "correct": right,
                    "opposite": opposite,
                    "reward": float(right) - 0.25 * opposite,
                }
            )
        results.append(observations)
    rewards = [
        (a["reward"] + b["reward"]) / 2 + 0.5 * (a["correct"] and b["correct"])
        for a, b in zip(*results, strict=True)
    ]
    return rewards, results


def token_objective(net, first, index, labels, temperature, cap):
    hidden = net.second_visit(first)
    logits = (
        net.project(net.text.model.norm(hidden[0, index])).astype(mx.float32)
        / temperature
    )
    nll = nn.losses.cross_entropy(logits, labels, reduction="none")
    return nll.sum() / cap, nll


def load_parent(net, root):
    parent = Path("research/results") / PARENT
    if not read(parent / "decision.json")["passed"]:
        raise ValueError("The Base warm-up did not qualify")
    file = parent / "reentry.safetensors"
    sha = digest(file)
    assert (
        sha
        == read(parent / "fit.json")["weights"]
        == read(parent / "checksums.json")[file.name]
    )
    before = mx.load(str(file))
    expected = dict(tree_flatten(net.trainable_parameters()))
    assert before.keys() == expected.keys() and all(
        x.shape == expected[k].shape and x.dtype == expected[k].dtype
        for k, x in before.items()
    )
    net.load_weights(list(before.items()), strict=False)
    save(
        root / "warm-parent.json",
        {"archive": digest(parent / "checksums.json"), "weights": sha},
    )
    return sha


def train(net, tok, s, root, guard):
    parent_sha = load_parent(net, root)
    world_list = sorted(
        read(SOURCE)["cohorts"]["train"],
        key=lambda w: fingerprint([s["fit_seed"], "policy", w["id"]]),
    )[: s["reward_updates"]]
    warm = inputs(tok, s)
    streams = decks(warm, s["fit_seed"] + 1)
    stream = streams["earlier"]
    lr = s["policy_lr"]
    opt = optimizer(lr, s)
    directory = Path("checkpoints/research") / NAME
    directory.mkdir(exist_ok=True, parents=True)
    identity = digest(root / "registration.json")
    cursor = {"step": 0, "registration": identity, "rollouts": []}
    if (directory / "latest.json").exists():
        cursor = checkpoint.restore(directory, net, opt)
        assert cursor["registration"] == identity
        for _ in range(cursor["step"]):
            next(stream)
    stop = set(tok.eos_token_ids) | {tok.convert_tokens_to_ids("<|im_end|>")}
    grad = nn.value_and_grad(net, token_objective)
    anchor_grad = nn.value_and_grad(net, gold_nll)
    select(net, "native")
    net.checkpointing = True
    net.checkpoint_policy = "span"
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        prepare_tables(folder, world_list)
        for step, w in enumerate(world_list, 1):
            if step <= cursor["step"]:
                continue
            guard.check()
            messages, defs, pairs = contextual(w, folder)
            phase = "query" if step % 3 == 0 else "answer"
            histories = [messages] if phase == "query" else pairs
            prefixes = [
                tok.encode(qwen_tools.render(tok, m, defs), add_special_tokens=False)
                for m in histories
            ]
            draws = [
                sample(
                    net,
                    p,
                    stop,
                    s["group_size"],
                    s["rollout_cap"],
                    s["temperature"],
                    s["fit_seed"] + step * 4 + j,
                    guard,
                )
                for j, p in enumerate(prefixes)
            ]
            rewards, details = scored(tok, w, folder, phase, draws, stop, defs)
            adv = advantages(rewards)
            net.train()
            gtotal = None
            drift = 0.0
            num = 0
            pg_norm = 0.0
            for batch, prefix in zip(draws, prefixes, strict=True):
                for row, weight in zip(batch, adv, strict=True):
                    if weight == 0:
                        continue
                    y = row["tokens"]
                    # Cache only the frozen ordinary branch. Gradients traverse all four extra applications.
                    net.eval()
                    h = net.first_visit(mx.array([prefix + y[:-1]]))
                    mx.eval(h)
                    net.train()
                    loss, g = grad(
                        net,
                        h,
                        mx.arange(len(y)) + len(prefix) - 1,
                        mx.array(y),
                        s["temperature"],
                        s["rollout_cap"],
                    )
                    gtotal = accumulate(
                        gtotal, g, weight / (s["group_size"] * len(histories))
                    )
                    mx.eval(gtotal, loss)
                    if not math.isfinite(loss[0].item()):
                        raise InterruptedError("Nonfinite on-policy parameter gradient")
                    drift += (
                        mx.abs(loss[1] + mx.array(row["behavior_logp"])).sum().item()
                    )
                    num += len(y)
            if gtotal is not None:
                pg_norm = mx.sqrt(
                    sum(mx.square(v).sum() for _, v in tree_flatten(gtotal))
                ).item()
            anchor = next(stream)
            _, prefix, y, at = warm[anchor]
            js = positions(y, at, s["warm_targets"], f"{s['fit_seed']}/anchor/{step}")
            net.eval()
            h = net.first_visit(mx.array([prefix + y[:-1]]))
            mx.eval(h)
            net.train()
            base_loss, g = anchor_grad(
                net, h, mx.array(js) + len(prefix) - 1, mx.array([y[j] for j in js])
            )
            gtotal = accumulate(gtotal, g, s["anchor_weight"])
            gn = apply(net, opt, gtotal)
            rec = {
                "step": step,
                "question": w["id"],
                "stage": phase,
                "prefixes": prefixes,
                "draws": draws,
                "rewards": rewards,
                "details": details,
                "advantages": adv,
                "policy_gradient_norm": pg_norm,
                "sampled_logp_mean_abs_replay_error": drift / num if num else None,
                "gradient_norm": gn,
                "anchor": anchor,
                "anchor_positions": js,
                "anchor_nll": base_loss.item(),
            }
            cursor["step"] = step
            cursor["rollouts"].append(rec)
            if step % 16 == 0 or step == s["reward_updates"]:
                checkpoint.write(directory, net, opt, cursor)
                print(
                    "Native execution-reward update",
                    step,
                    "current rewards",
                    rewards,
                    flush=True,
                )
    weights = root / "reentry.safetensors"
    mx.save_safetensors(str(weights), dict(tree_flatten(net.trainable_parameters())))
    shutil.copyfile(
        checkpoint.location(directory) / "optimizer.safetensors",
        root / "optimizer.safetensors",
    )
    packed(root / "on-policy-rollouts.json.gz", cursor["rollouts"])
    evidence = parity(
        net, read(Path("research/results") / PARENT / "initial-parity.json")["ids"]
    )
    save(root / "trained-parity.json", evidence)
    save(
        root / "fit.json",
        {
            "weights": digest(weights),
            "registration": identity,
            "updates": cursor["step"],
            "parent": parent_sha,
            "trainable": sum(
                v.size for _, v in tree_flatten(net.trainable_parameters())
            ),
        },
    )
    return outcome(cursor["rollouts"], evidence, parent_sha != digest(weights))


def outcome(log, parity_report, changed):
    c = Counter()
    for row in log:
        c["unequal_returns"] += len(set(row["rewards"])) > 1
        c["nonzero_policy_gradient"] += (
            math.isfinite(row["policy_gradient_norm"])
            and row["policy_gradient_norm"] > 0
        )
        if row["stage"] == "query":
            c["groups_with_correct"] += any(
                y["correct"] for r in row["details"] for y in r["results"]
            )
        else:
            c["groups_with_correct"] += any(
                r["correct"] for world in row["details"] for r in world
            )
        c["sampled_tokens"] += sum(
            len(r["tokens"]) for world in row["draws"] for r in world
        )
    checks = {
        "varied": c["unequal_returns"] >= 16,
        "correct": c["groups_with_correct"] >= 8,
        "policy_gradient": c["nonzero_policy_gradient"] >= 16,
        "changed": changed,
        "independent": parity_report["sites"] == parity_report["unique_slots"] == 28
        and parity_report["full_top"] == parity_report["incremental_top"],
    }
    return {
        "passed": all(checks.values()),
        "gates": checks,
        "measured": dict(c),
        "training_precondition_only": True,
    }


def verify(root, d):
    if d.get("status") == "interrupted":
        assert not d["passed"] and d["reason"]
        return
    s = registered(NAME)
    tok = tokenizer(s)
    assert s == read(root / "registration.json")
    world_list = sorted(
        read(SOURCE)["cohorts"]["train"],
        key=lambda w: fingerprint([s["fit_seed"], "policy", w["id"]]),
    )[: s["reward_updates"]]
    stop = set(tok.eos_token_ids) | {tok.convert_tokens_to_ids("<|im_end|>")}
    log = unpack(root / "on-policy-rollouts.json.gz")
    assert (
        len(log) == s["reward_updates"]
        and digest(root / "reentry.safetensors") == read(root / "fit.json")["weights"]
    )
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        prepare_tables(folder, world_list)
        for step, (w, row) in enumerate(zip(world_list, log, strict=True), 1):
            assert step == row["step"] and w["id"] == row["question"]
            msgs, defs, pairs = contextual(w, folder)
            phase = "query" if step % 3 == 0 else "answer"
            assert row["stage"] == phase and row["prefixes"] == [
                tok.encode(qwen_tools.render(tok, m, defs), add_special_tokens=False)
                for m in ([msgs] if phase == "query" else pairs)
            ]
            for draws in row["draws"]:
                assert len(draws) == s["group_size"]
                for r in draws:
                    assert (
                        0
                        < len(r["tokens"])
                        == len(r["behavior_logp"])
                        <= s["rollout_cap"]
                    )
                    assert all(
                        math.isfinite(lp) and lp <= 1e-5 for lp in r["behavior_logp"]
                    )
                    assert all(i not in stop for i in r["tokens"][:-1]) and (
                        r["tokens"][-1] in stop or len(r["tokens"]) == s["rollout_cap"]
                    )
            rv, details = scored(tok, w, folder, phase, row["draws"], stop, defs)
            assert (
                rv == row["rewards"]
                and details == row["details"]
                and advantages(rv) == row["advantages"]
            )
    assert d == outcome(
        log,
        read(root / "trained-parity.json"),
        read(root / "fit.json")["parent"] != digest(root / "reentry.safetensors"),
    )


def run():
    s = registered(NAME)
    root = root_for(s)
    capture(root, (*CODES, "loopqwen/research/table_policy_learning.py"))
    guard = Budget(NAME, s["limits"])
    try:
        d = train(from_base(s["checkpoint"], s), tokenizer(s), s, root, guard)
    except (TimeoutError, MemoryError, InterruptedError) as exc:
        d = {"passed": False, "status": "interrupted", "reason": str(exc)}
    finally:
        guard.close()
    save(root / "decision.json", d)


if __name__ == "__main__":
    run()
