"""Resume only a complete prefix of the registered alternating two-arm schedule."""

from ..evaluation.mechanism import unpack
from ..identity import fingerprint
from ..io import read, save


def resume(root, spec, items, sides):
    prompts = [fingerprint(w["prompt_ids"]) for w in items]
    identity = {
        "registration": fingerprint(spec),
        "prompts": prompts,
        "arms": list(sides),
    }
    meta = root / "progress-identity.json"
    files = {s: root / (s + ".json.gz") for s in sides}
    if meta.exists():
        if read(meta) != identity:
            raise ValueError(
                "Existing crossover uses a different registration, order or input"
            )
    elif any(f.exists() for f in files.values()):
        raise ValueError("Existing crossover is missing its measurement identity")
    else:
        save(meta, identity)
    records = {s: unpack(f) if f.exists() else [] for s, f in files.items()}
    for arm in records.values():
        if [r["input_sha256"] for r in arm] != prompts[: len(arm)]:
            raise ValueError("Saved measurement differs from the frozen input")
    gap = False
    for i in range(len(items)):
        for side in sides if i % 2 == 0 else reversed(sides):
            if len(records[side]) <= i:
                gap = True
            elif gap:
                raise ValueError(
                    "Existing results are not a prefix of the alternating measurement order"
                )
    return records
