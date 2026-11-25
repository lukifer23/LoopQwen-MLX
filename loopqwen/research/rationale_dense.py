"""Finish the matched-data K1 attribution for the passing rationale deployment."""

from pathlib import Path

from .. import checkpoint
from ..evaluation.mechanism import unpack
from ..io import checkpoint_location, digest, read, save
from . import bootstrap, core, rationale

REG = Path("research/experiments/rationale-dense-01.json")


def run():
    reg = read(REG)
    prior = Path(reg["prerequisite"])
    earlier = read(prior / "decision.json")
    if (
        not earlier["passed"]
        or digest(prior / "model/adapters.safetensors") != reg["recurrent_weights"]
    ):
        raise ValueError("Wrong recurrent comparison")
    original = read(rationale.REG)
    docs, items, manifest = rationale.material(original)
    frozen = Path("research/data") / (original["id"] + ".json")
    if manifest != read(frozen) or earlier["data"] != digest(frozen):
        raise ValueError("Unmatched training exposure or prompt set")
    if digest(Path(reg["parent"]) / "adapters.safetensors") != reg["parent_sha256"]:
        raise ValueError("Wrong matched dense parent")
    out = core.OUT / reg["id"]
    if (out / "decision.json").exists():
        raise FileExistsError(out)
    core.assets()
    b = core.Budget(reg["id"], reg["limits"])
    result = {"registration": digest(REG), "data": digest(frozen), "passed": False}
    cp = Path("checkpoints/research") / reg["id"]
    try:
        m, t, st = checkpoint.load_trained(reg["parent"])
        if m.span.passes != 1:
            raise ValueError("Expected ordinary-depth model")
        m.checkpointing = True
        spec = {
            k: original[k]
            for k in (
                "updates",
                "logical_batch",
                "cap_positions",
                "lr",
                "clip",
                "weight_decay",
                "generation_cap",
            )
        }
        spec.update(
            id=reg["id"],
            depth=1,
            deployment={"config": st["config"], "alpha_override": m.alpha_override},
        )
        state = core.train(m, docs, spec, cp, b, reg["parent_sha256"])
        gen = checkpoint_location(cp)
        weight = digest(gen / "adapters.safetensors")
        bootstrap.release(gen, out / "model")
        result.update(
            weights=weight,
            updates=state["step"],
            update_seconds=sum(h["seconds"] for h in state["history"]),
            targets=sum(h["targets"] for h in state["history"]),
        )
        if result["targets"] != earlier["targets"]:
            raise ValueError("Unmatched supervised target count")
        m.eval()
        dense = rationale.assess(m, t, weight, items, spec, b, out / "dense.json.gz")
        dense["retention"] = bootstrap.retention(m, 1, b)
        loop = earlier["candidate"]
        a = unpack(out / "dense.json.gz")["records"]
        c = unpack(prior / "candidate.json.gz")["records"]
        if [r["prompt_ids"] for r in a] != [r["prompt_ids"] for r in c]:
            raise ValueError("Unmatched deployed prompts")
        differences = [
            int(y["correct"]) - int(x["correct"]) for x, y in zip(a, c, strict=True)
        ]
        result.update(
            status="completed",
            dense=dense,
            recurrent=loop,
            recurrent_weights=reg["recurrent_weights"],
            wins=differences.count(1),
            losses=differences.count(-1),
            passed=sum(differences) >= 2
            and loop["seconds"] <= 1.25 * dense["seconds"]
            and loop["retention"]["nll"] <= dense["retention"]["nll"] + 0.03,
        )
    except (MemoryError, InterruptedError, TimeoutError) as exc:
        result.update(status="interrupted", reason=str(exc))
    finally:
        b.close()
    save(out / "decision.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    run()
