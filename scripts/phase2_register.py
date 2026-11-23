"""Register the data/build identity and explicit protocol before scoring."""

from pathlib import Path

from loopqwen import REVISION
from loopqwen.data.build_data import digest
from loopqwen.io import read, save


def register(name, runs, purpose, matched=(), final=False):
    path = Path("experiments") / (name + ".json")
    m = dict(
        runs=runs,
        purpose=purpose,
        matched=list(matched),
        revision=REVISION,
        data_sha256=digest(Path("data/v2/manifest.json")),
        splits=["test", "ood"] if final else ["validation"],
        loss_count=150 if final else 90,
        generation={"answer": 12, "trace": 128},
        generation_count={"answer": 90 if final else 30, "trace": 15 if final else 5},
        language_count=48 if final else 24,
    )
    if path.exists() and read(path) != m:
        raise ValueError("Choose a fresh experiment ID")
    save(path, m)


if __name__ == "__main__":
    register(
        "phase2_direct",
        ["p2_a_" + p + "_" + v for p in ("answer", "trace") for v in ("dense", "loop")],
        "Stage A development comparison of answer-only and verified-trace supervision.",
        [[f"p2_a_{p}_dense", f"p2_a_{p}_loop"] for p in ("answer", "trace")],
    )
