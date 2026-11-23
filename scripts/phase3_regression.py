"""Run the native suite in an isolated output directory to preserve historical measurements."""

import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from loopqwen.io import save

out = root / "outputs/phase3"
cwd = out / "regression"
(cwd / "outputs").mkdir(exist_ok=True, parents=True)
for name in ("data", "adapters", "configs", "docs", "experiments", "checkpoints"):
    link = cwd / name
    if not link.exists():
        link.symlink_to(root / name, target_is_directory=True)
start = perf_counter()
cmd = [sys.executable, "-m", "pytest", "-q", str(root / "tests")]
with (out / "regression.log").open("w") as log:
    subprocess.run(
        cmd,
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(root)},
        stdout=log,
        stderr=subprocess.STDOUT,
        check=True,
    )
summary = (out / "regression.log").read_text().splitlines()[-1]
save(
    out / "regression.json",
    dict(
        summary=summary,
        seconds=perf_counter() - start,
        command="python -m pytest -q tests",
    ),
)
print(summary, flush=True)
