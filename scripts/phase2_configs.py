"""Materialize independent runs. All arms share the same pinned repo defaults."""

from pathlib import Path

import yaml

ROOT = Path("configs/phase2")


def write(name, **kwargs):
    c = dict(
        name="p2_" + name,
        dataset="v2",
        mixture="verified_traces",
        sequence_length=256,
        batch_size=4,
        accumulation=1,
        steps=72,
        warmup=5,
        checkpoint_interval=24,
        eval_interval=36,
        eval_count=90,
        memory_target_gib=8,
        min_available_gib=1,
        gradient_checkpointing=True,
        seed=41,
    )
    c.update(kwargs)
    path = ROOT / (c["name"] + ".yaml")
    path.write_text(yaml.safe_dump(c, sort_keys=False))
    return path


if __name__ == "__main__":
    ROOT.mkdir(exist_ok=True, parents=True)
    for mix, label in [("answer_only", "answer"), ("verified_traces", "trace")]:
        for k, arm in [(1, "dense"), (2, "loop")]:
            write(f"a_{label}_{arm}", mixture=mix, passes=k)
