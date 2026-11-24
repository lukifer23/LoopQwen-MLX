"""Two deployed models on identical chat prompts with unrestricted greedy decoding."""

import gc
import json
from pathlib import Path

import mlx.core as mx

from .. import REVISION
from ..data import pointer5, pointer6
from ..identity import fingerprint
from ..io import digest, read, save
from ..looped_model import LoopedModel
from ..model_loader import load_base
from ..recurrence import Span
from ..training.mechanism import StudyArm, assets
from .answers import numeric
from .generation_eval import generate
from .mechanism import load, packed, unpack

REG = Path("experiments/phase6_reasoning_pair.json")


def questions(seed):
    key = lambda r: fingerprint([seed, r["id"]])
    pointers = []
    for row in pointer5.rows("development"):
        r = pointer6.ordered(row, "random0", 61091)
        edges = ", ".join(f"{i} goes to {r['map'][i]}" for i in r["order"])
        pointers.append(
            {
                "id": r["example_id"],
                "family": "pointer",
                "split": "permutation",
                "gold": str(r["states"][2]),
                "question": f"The following links connect eight rooms: {edges}. Starting in room {r['start']}, follow exactly three links. Which room do you reach?",
            }
        )
    procedural = []
    for split in ("development", "development_ood"):
        data = [
            json.loads(line)
            for line in (Path("data/v3") / f"{split}.jsonl").read_bytes().splitlines()
        ]
        for family in sorted({r["category"] for r in data}):
            r = min((r for r in data if r["category"] == family), key=key)
            procedural.append(
                {
                    "id": r["id"],
                    "family": family,
                    "split": split,
                    "question": r["question"],
                    "gold": str(r["answer"]),
                }
            )
    return [
        r
        for pair in zip(sorted(pointers, key=key)[:10], procedural, strict=True)
        for r in pair
    ]


def score(text):
    if "</think>" not in text:
        return None
    visible = text.rsplit("</think>", 1)[1].split("<|im_end|>")[0]
    return numeric(visible)


def run():
    spec = read(REG)
    rows = questions(spec["selection_seed"])
    arm = StudyArm.registered("randomized", 109)
    dest = Path("outputs/phase6/109") / spec["id"]
    dest.mkdir(exist_ok=True, parents=True)

    def measure(name, cohort):
        if name == "base":
            base, t, _ = load_base()
            m = LoopedModel(base, Span(passes=1))
            w = assets()["files"]["model.safetensors-00001-of-00001.safetensors"]
            del base
        else:
            m, t, _, w = load(arm, 480)
        if m.span.passes != (1 if name == "base" else 3):
            raise ValueError("Deployment depth mismatch")
        m.eval()
        generate(m, t, "1 + 1 =", 1)
        for r in cohort:
            message = (
                r["question"]
                + "\nGive your final answer on a line of its own as Final answer: <number>."
            )
            prompt = t.apply_chat_template(
                [{"role": "user", "content": message}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            if not prompt.rstrip().endswith("<think>"):
                raise ValueError("Tokenizer did not open the thinking segment")
            ids = t.encode(prompt, add_special_tokens=False)
            path = dest / f"{name}-{r['id']}.json.gz"
            identity = {
                "arm": name,
                "weights": w,
                "base": REVISION,
                "assets": fingerprint(assets()),
                "registration": digest(REG),
                "question": fingerprint(r),
                "inputs": fingerprint(ids),
                "source": digest(Path(__file__)),
                "scorer": digest(Path("loopqwen/evaluation/answers.py")),
                "generation": digest(Path("loopqwen/evaluation/generation_eval.py")),
            }
            if path.exists():
                if unpack(path)["identity"] != identity:
                    raise ValueError(f"Stale paired record at {path}")
                continue
            trial = generate(m, t, prompt, spec["max_tokens"])
            answer = score(trial["text"])
            eos = trial["token_ids"][-1] in t.eos_token_ids
            packed(
                path,
                {
                    "identity": identity,
                    "item": r,
                    "prompt_ids": ids,
                    "generation": trial,
                    "answer": answer,
                    "correct": answer == numeric(r["gold"]),
                    "stop": "eos" if eos else "cap",
                    "thinking_closed": "</think>" in trial["text"],
                },
            )
        del m, t
        gc.collect()
        mx.clear_cache()

    completed = []
    for block in (rows[:6], rows[6:]):
        half = len(block) // 2
        measure("base", block[:half])
        measure("recurrent", block)
        measure("base", block[half:])
        completed.extend(block)
        if len(completed) == 6:
            pilot = [
                unpack(dest / f"{name}-{r['id']}.json.gz")
                for r in completed
                for name in ("base", "recurrent")
            ]
            if all(x["answer"] is None and x["stop"] == "cap" for x in pilot):
                break
    rows = completed
    records = []
    for r in rows:
        b, l = [unpack(dest / f"{n}-{r['id']}.json.gz") for n in ("base", "recurrent")]
        if b["prompt_ids"] != l["prompt_ids"] or b["item"] != l["item"]:
            raise ValueError("Unpaired prompts")
        records.append((b, l))

    def stats(group):
        out = {}
        for i, name in enumerate(("base", "recurrent")):
            xs = [pair[i] for pair in group]
            out[name] = {
                "correct": sum(x["correct"] for x in xs),
                "valid": sum(x["answer"] is not None for x in xs),
                "capped": sum(x["stop"] == "cap" for x in xs),
                "seconds": sum(x["generation"]["elapsed_s"] for x in xs),
                "tokens": sum(len(x["generation"]["token_ids"]) for x in xs),
            }
        out.update(
            n=len(group),
            base_only=sum(b["correct"] and not l["correct"] for b, l in group),
            loop_only=sum(l["correct"] and not b["correct"] for b, l in group),
        )
        return out

    result = {
        k: stats([p for p in records if k == "all" or p[0]["item"]["split"] == k])
        for k in ("all", "permutation", "development", "development_ood")
    }
    save(dest / "scores.json", result)
    print(result, flush=True)
