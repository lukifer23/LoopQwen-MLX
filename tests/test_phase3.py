import random
import subprocess
import sys
from pathlib import Path


from loopqwen.data.graphs import graph, solve
from loopqwen.evaluation.trace_verifier import verify


def test_trace_certificates():
    r = dict(answer="12", trace="7 + 5 = 12\nFinal answer: 12")
    assert verify(r, r["trace"])["verified"]
    wrong = verify(r, "7 + 5 = 11\nFinal answer: 12")
    assert wrong["final"] == "correct" and wrong["checks"] == ["arithmetic_error"]
    assert verify(r, "A dozen.\nFinal answer: 12")["checks"] == ["unsupported"]
    assert (
        verify(r, "Final answer: 11\nFinal answer: 12")["final"]
        == "conflicting_final_answers"
    )
    assert verify(r, "7 + 5 =", True)["truncated"]
    r = dict(answer="5", trace="Sorted: [2, 3]\n2 + 3 = 5\nFinal answer: 5")
    assert (
        verify(r, r["trace"].replace("[2, 3]", "[3, 2]"))["checks"][0]
        == "incorrect_state_transition"
    )


def test_graph_paths_and_fingerprints():
    seen = set()
    for seed in range(90):
        r = graph(
            random.Random(seed), seed % 3, ("id", "operand", "topology")[seed % 3]
        )
        d = r["difficulty"]
        g = r["graph"]
        assert d["roots"] >= 2 and d["sinks"] >= 2 and d["joins"] >= 1
        assert (
            str(solve(g["durations"], g["predecessors"], g["target"])[0][g["target"]])
            == r["answer"]
        )
        assert r["id"] not in seen and verify(r, r["trace"])["verified"]
        seen.add(r["id"])


def test_checkout_contention(tmp_path):
    import fcntl
    import os
    import shutil
    from loopqwen.worker import lease_path

    repo = Path(__file__).resolve().parents[1]
    other = tmp_path / "second-checkout"
    pkg = other / "loopqwen"
    pkg.mkdir(parents=True)
    for name in ("__init__.py", "worker.py"):
        shutil.copyfile(repo / "loopqwen" / name, pkg / name)

    env = dict(
        os.environ,
        HOME=str(tmp_path),
        PYTHONPATH=str(other),
    )
    path = tmp_path / lease_path().relative_to(Path.home())
    path.parent.mkdir(parents=True)
    with path.open("w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        result = subprocess.run(
            [sys.executable, "-c", "from loopqwen.worker import acquire; acquire()"],
            cwd=other,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode and "user-wide GPU lease" in result.stderr


def test_evaluation_revision_tracks_decoding_and_metadata(tmp_path):
    from loopqwen.io import checkpoint_location, read, save
    from loopqwen.evaluation.identity import evaluation_identity

    source = checkpoint_location("adapters/p2_confirm_43_loop")
    # Use the published adapter tensors; the metadata digest also includes non-weight changes.
    (tmp_path / "adapters.safetensors").symlink_to(
        (source / "adapters.safetensors").resolve()
    )
    state = read(source / "state.json")
    if "manifest" not in state:
        state["manifest"] = read("data/v2/manifest.json")
    save(tmp_path / "state.json", state)

    def identity(cap):
        return evaluation_identity(
            {"dataset": "v2"}, tmp_path, ["item"], {"cap": cap}, "v3"
        )

    original = identity(192)
    assert original != identity(128)
    state["metric_protocol"] = "new"
    save(tmp_path / "state.json", state)
    assert original != identity(192)
    assert original["weights"] == identity(192)["weights"]


def test_dag_certificate_accepts_independent_order_and_omitted_irrelevant_sink():
    row = graph(random.Random(75), 1)
    g = row["graph"]
    required = {g["target"]}
    for _ in g["durations"]:
        required.update(p for node in tuple(required) for p in g["predecessors"][node])
    claims = [
        line for line in row["trace"].splitlines()[:-1] if line.split()[1] in required
    ]
    text = "\n".join(list(reversed(claims)) + ["Final answer: " + row["answer"]])
    assert verify(row, text)["verified"]
    assert not verify(row, "\n".join(text.splitlines()[1:]))["verified"]


def test_active_lease_survives_cache_eviction(tmp_path):
    import fcntl
    from loopqwen.worker import _lock

    stable = tmp_path / "application-support" / "gpu.lock"
    cache = tmp_path / "caches" / "gpu.lock"
    with _lock(stable, None), _lock(cache, None):
        cache.unlink()
        # A different process can open a new cache inode after a cleaner ran.
        # The persistent lease must prevent it from starting a second model.
        result = subprocess.run(
            [sys.executable, "-c", "import fcntl,sys; f=open(sys.argv[1]); "
             "fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)", str(stable)],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode and "BlockingIOError" in result.stderr
    with stable.open() as file:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
