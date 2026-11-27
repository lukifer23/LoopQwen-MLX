"""Learn emitted explanations and verdicts on a shared recurrent instruction layer."""

import gc
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten, tree_map
from mlx_lm import batch_generate
from mlx_lm.utils import load_model
from tokenizers import Tokenizer

from .. import checkpoint
from ..evaluation.mechanism import packed, unpack
from ..export_runtime import load_export_tokenizer
from ..identity import fingerprint
from ..io import digest, read, save
from ..model_loader import count
from ..qwen_tools import assistant_message
from .file_evidence_fit import pack, projected
from .reasoned_evidence_data import bind_file, table, text

LABELS = ("true", "false", "uncertain")


def parse_teacher(raw, n):
    clean = raw.replace("**", "").strip()
    match = re.search(
        r"(?:^|\n)\s*(?:final\s+)?verdict\s*:\s*(true|false|uncertain)\s*[.!]?\s*$",
        clean,
        re.IGNORECASE,
    )
    if not match:
        return None
    proof = re.sub(
        r"^\s*(?:explanation|reasoning|reason)\s*:\s*",
        "",
        clean[: match.start()].strip(),
        flags=re.IGNORECASE,
    )
    refs = [
        int(x)
        for k in re.findall(r"\[([\d,\s]+)\]", proof)
        for x in re.findall(r"\d+", k)
    ]
    refs.extend(
        int(x) for x in re.findall(r"\bpremise\s+(\d+)\b", proof, re.IGNORECASE)
    )
    if not (5 <= len(proof.split()) <= 100 and refs and all(1 <= i <= n for i in refs)):
        return None
    return {"reason": proof, "label": match.group(1).lower(), "cited": refs}


def teach(s, tok, root, guard):
    cfg = s["teacher"]
    directory = Path(cfg["directory"])
    data = read(s["training"])["teacher_pool"]
    assert digest(directory / "model.safetensors") == cfg["weights_sha256"]
    assert all(digest(directory / k) == v for k, v in cfg["files"].items())
    assert (
        Tokenizer.from_file(str(directory / "tokenizer.json")).get_vocab()
        == tok.get_vocab()
    )
    start = perf_counter()
    path = root / "teacher-generations.json.gz"
    saved = unpack(path) if path.exists() else []
    assert [r["id"] for r in saved] == [r["id"] for r in data[: len(saved)]]
    if len(saved) < len(data):
        generation_tok = load_export_tokenizer(
            Path(s["checkpoint"]),
            eos_token_ids=[*tok.eos_token_ids, tok.convert_tokens_to_ids("<|im_end|>")],
        )
        model, _ = load_model(directory)
        model.eval()
        for i in range(len(saved), len(data), s["teacher_batch"]):
            guard.check()
            group = data[i : i + s["teacher_batch"]]
            prompts = []
            for row in group:
                prompt = (
                    row["evidence"]
                    + "\n\nClaim: "
                    + row["claim"]
                    + "\nAssess only these numbered premises. Give a short explanation citing premises as [1], [2], etc. Use true if the claim follows, false if its negation follows, otherwise uncertain. End with a separate line 'Verdict: true', 'Verdict: false', or 'Verdict: uncertain'."
                )
                wire = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    enable_thinking=False,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                prompts.append(tok.encode(wire, add_special_tokens=False))
            reply = batch_generate(
                model.language_model,
                generation_tok,
                prompts,
                max_tokens=s["teacher_cap"],
                return_token_ids=True,
                verbose=False,
            )
            for row, ids, gen in zip(group, prompts, reply.token_ids, strict=True):
                raw = tok.decode(gen)
                candidate = parse_teacher(raw, len(row["evidence"].splitlines()))
                saved.append(
                    {
                        "id": row["id"],
                        "prompt": ids,
                        "generated": gen,
                        "text": raw,
                        "parsed": candidate,
                        "accepted": bool(
                            len(gen) < s["teacher_cap"]
                            and candidate
                            and candidate["label"] == row["label"]
                        ),
                    }
                )
            packed(path, saved)
            if len(saved) % 32 == 0:
                print(
                    "Reasoning teacher",
                    len(saved),
                    sum(r["accepted"] for r in saved),
                    flush=True,
                )
        del model
        gc.collect()
        mx.clear_cache()
    counts = Counter(r["parsed"]["label"] for r in saved if r["accepted"])
    rep = root / "teacher-fit.json"
    if not rep.exists():
        save(
            rep,
            {
                "targets": digest(path),
                "training": digest(Path(s["training"])),
                "weights": cfg["weights_sha256"],
                "accepted": dict(counts),
                "checked": len(saved),
                "seconds": perf_counter() - start,
                "peak": mx.get_peak_memory(),
            },
        )
    assert read(rep)["targets"] == digest(path)
    if (
        len(saved) != len(data)
        or sum(counts.values()) < s["teacher_min_total"]
        or any(counts[l] < s["teacher_min_per_label"] for l in LABELS)
    ):
        raise InterruptedError(
            "Independent explanation teacher missed its registered correct-verdict and citation coverage"
        )


def turns(s, root):
    data = read(s["training"])
    by_id = {r["id"]: r for r in data["teacher_pool"]}
    supervised = [("human", w) for w in data["human"]]
    for result in unpack(root / "teacher-generations.json.gz"):
        if result["accepted"]:
            supervised.append(
                (
                    "teacher",
                    {**by_id[result["id"]], "reason": result["parsed"]["reason"]},
                )
            )
    return supervised_turns(supervised)


def supervised_turns(supervised):
    examples = []
    with TemporaryDirectory() as temp:
        tmp = Path(temp)
        for i, (source, r) in enumerate(supervised):
            key = "Record " + fingerprint(r["id"])[:8]
            other = [supervised[(i + j) % len(supervised)][1] for j in (1, 2)]
            filename = ["case-file.csv", "notes.csv"][i % 2]
            raw = table(
                ["record", "observations"],
                [
                    (k, v)
                    for k, v in sorted(
                        [
                            (key, r["evidence"]),
                            *(
                                ("Record " + fingerprint(w["id"])[:8], w["evidence"])
                                for w in other
                            ),
                        ]
                    )
                ],
            )
            request = text(f'Read "{key}" from the attached file.', r["claim"])
            if source == "package":
                peers = {w["package"]: w for src, w in supervised if src == source}
                key = r["package"]
                filename = "distribution-metadata.csv"
                raw = table(
                    ["Name", "Requires-Python", "Version"],
                    [
                        (w["package"], w["requirement"], w["release"])
                        for w in sorted(peers.values(), key=lambda x: x["package"])
                    ],
                )
                request = text(
                    f'Look up the Python requirement for "{key}" in the metadata file.',
                    r["claim"],
                )
            defs, call, _, hist = bind_file(
                tmp, filename, raw, key, r["evidence"], request
            )
            answer = {
                "role": "assistant",
                "content": r["label"],
                "reasoning_content": r["reason"],
            }
            for kind, decl, msgs, tgt in (
                (
                    "inline",
                    [],
                    [{"role": "user", "content": text(r["evidence"], r["claim"])}],
                    answer,
                ),
                ("file", defs, hist, answer),
                ("opening", defs, hist[:1], assistant_message([call])),
            ):
                examples.append(
                    {
                        "id": r["id"] + "-" + kind,
                        "source": source,
                        "case": r["id"],
                        "label": r["label"],
                        "kind": kind,
                        "tools": decl,
                        "messages": msgs,
                        "target": tgt,
                    }
                )
    old = read("research/data/cross-view-training.json")
    for row in old["examples"]:
        if row["kind"] == "rehearsal":
            examples.append(row)
    return examples


def fit(
    net,
    tok,
    s,
    root,
    guard,
    *,
    prepare,
    make_views=turns,
    draw_cases=None,
    refinement=None,
):
    start = perf_counter()
    examples = make_views(s, root)
    packed(root / "training-views.json.gz", examples)
    bank = prepare(net, tok, {"examples": examples}, guard)
    sources = defaultdict(lambda: defaultdict(dict))
    rehearsal = []
    for row, sample in zip(examples, bank, strict=True):
        if row["kind"] == "rehearsal":
            rehearsal.append(sample)
            continue
        sources[row["source"]][row["label"]].setdefault(row["case"], {})[
            row["kind"]
        ] = sample
    streams = {
        k: {l: list(v.values()) for l, v in by_label.items()}
        for k, by_label in sources.items()
    }
    rng = random.Random(s["fit_seed"])

    def draw():
        if draw_cases is not None:
            return draw_cases(rng, streams, rehearsal)
        pair = [
            rng.choice(streams[k][rng.choice(LABELS)]) for k in ("human", "teacher")
        ]
        return pair, rng.choice(pair)["opening"], rng.choice(rehearsal)

    end_think = tok.convert_tokens_to_ids("</think>")

    def judged(model, h, r, c, y, reason_positions, answer_positions):
        ce = nn.losses.cross_entropy(projected(model, h, r, c), y, reduction="none")
        return (
            0.50 * ce[reason_positions].mean()
            + 0.35 * ce[answer_positions].mean()
            + 0.15 * ce.mean()
        )

    def ordinary(model, h, r, c, y):
        return nn.losses.cross_entropy(projected(model, h, r, c), y).mean()

    raw = nn.value_and_grad(net, ordinary)
    reason = nn.value_and_grad(net, refinement(tok) if refinement else judged)
    net.train()
    net.checkpointing = True
    adam = optim.AdamW(learning_rate=s["lr"], weight_decay=0.01)
    prefix_seconds = perf_counter() - start
    folder = Path("checkpoints/research") / s["id"]
    folder.mkdir(parents=True, exist_ok=True)
    binding = digest(root / "registration.json")
    history = []
    previous = total = proof_tokens = 0
    prior_time = 0.0
    if (folder / "latest.json").exists():
        state = checkpoint.restore(folder, net, adam)
        assert state["registration"] == binding and state["views"] == digest(
            root / "training-views.json.gz"
        )
        previous, total, proof_tokens, history, prior_time = (
            state[k] for k in ("step", "tokens", "proof_tokens", "loss", "seconds")
        )
        for _ in range(previous):
            draw()
    start = perf_counter()
    for step in range(previous, s["updates"]):
        guard.check()
        pair, opening, other = draw()
        grads = None
        objective = 0.0
        for batch, weight in (
            ([pair[0]["inline"], pair[0]["file"]], 0.325),
            ([pair[1]["inline"], pair[1]["file"]], 0.325),
            ([opening], 0.20),
            ([other], 0.15),
        ):
            if len(batch) == 2:
                labels = batch[0][2]
                assert labels == batch[1][2]
                cut = labels.index(end_think)
                rp = [j for j in range(cut) if tok.decode([labels[j]]).strip()]
                ap = [
                    j
                    for j in range(cut + 1, len(labels) - 1)
                    if tok.decode([labels[j]]).strip()
                ]
                assert rp and ap
                if refinement:
                    row_label = tok.decode(labels[ap[0] : ap[-1] + 1]).strip()
                    assert (
                        row_label in LABELS
                        and labels[ap[0]]
                        == tok.encode(row_label, add_special_tokens=False)[0]
                    )
                pp, aa = [
                    mx.array([*ix, *(j + len(labels) for j in ix)]) for ix in (rp, ap)
                ]
                loss, g = reason(net, *pack(batch), pp, aa)
                proof_tokens += len(rp) * 2
            else:
                loss, g = raw(net, *pack(batch))
            grads = (
                tree_map(lambda x, scale=weight: x.astype(mx.float32) * scale, g)
                if grads is None
                else tree_map(
                    lambda a, b, scale=weight: a + b.astype(mx.float32) * scale,
                    grads,
                    g,
                )
            )
            mx.eval(grads, loss)
            objective += loss.item() * weight
            total += sum(len(item[2]) for item in batch)
        if not math.isfinite(objective):
            raise InterruptedError("Nonfinite reasoning objective")
        adam.update(net, grads)
        mx.eval(net.trainable_parameters(), adam.state)
        guard.check()
        if step == 0 or (step + 1) % 100 == 0:
            history.append({"step": step + 1, "loss": objective})
            print("Reasoning fit", history[-1], flush=True)
            if step:
                checkpoint.write(
                    folder,
                    net,
                    adam,
                    {
                        "registration": binding,
                        "views": digest(root / "training-views.json.gz"),
                        "step": step + 1,
                        "tokens": total,
                        "proof_tokens": proof_tokens,
                        "loss": history,
                        "seconds": prior_time + perf_counter() - start,
                    },
                )
    dest = root / "tool-adapters.safetensors"
    mx.save_safetensors(str(dest), dict(tree_flatten(net.trainable_parameters())))
    save(
        root / "fit.json",
        {
            "weights": digest(dest),
            "training": digest(Path(s["training"])),
            "views": digest(root / "training-views.json.gz"),
            "trainable": count(net, True),
            "passes": 2,
            "updates": s["updates"],
            "targets": total,
            "explanation_targets": proof_tokens,
            "turns": len(bank),
            "prepare_seconds": prefix_seconds,
            "optimizer_seconds": prior_time + perf_counter() - start,
            "peak": mx.get_peak_memory(),
            "loss": history,
        },
    )
