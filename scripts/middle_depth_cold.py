"""Cold replay two public turns on a relocated, offline installation and real provider."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep

from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.io import digest, read
from loopqwen.worker import child_lease
from scripts.middle_depth_archive import ARCHIVE, verify
from scripts.middle_depth_budget import ROOT, worker
from scripts.middle_depth_development import SOURCE, instance
from scripts.verify_runtime_wheel import CODE, trace

GUARD = """import sys
def isolate(event, args):
    if event in ("socket.connect", "socket.getaddrinfo"):
        raise AssertionError("Cold verification attempted network access")
sys.addaudithook(isolate)
import atexit, json, os, signal, threading, time
max_native_peak = 0
def report_mlx():
    global max_native_peak
    core = sys.modules.get("mlx.core")
    get_peak = getattr(core, "get_peak_memory", None)
    if get_peak is None:
        return
    max_native_peak = max(max_native_peak, get_peak())
    peak = max_native_peak
    name = os.environ["LOOPQWEN_COLD_METRICS"]
    with open(name + ".tmp", "w") as out:
        json.dump({"peak_bytes": peak, "seconds": time.monotonic()}, out)
    os.replace(name + ".tmp", name)
    if peak >= 8 * 2**30:
        os.kill(os.getpid(), signal.SIGTERM)
def watch_native():
    while not halted.wait(0.25):
        report_mlx()
def close_metrics():
    halted.set()
    thread.join()
    report_mlx()
halted = threading.Event()
thread = threading.Thread(target=watch_native, daemon=True)
thread.start()
atexit.register(close_metrics)
"""


def cold(stage, source, cap, archive_stage, arm):
    if source == "anne":
        from scripts.middle_depth_documents import SPECS

        task = read(SPECS / "anne-dev.json")
        folder = ROOT / "documents" / stage / "anne" / arm / str(cap)
        names = ["original-first", "original-own"]
    else:
        task = next(r for r in read(SOURCE) if r["id"] == source)
        folder = ROOT / "short" / stage / str(cap) / arm
        names = [source + "-" + s for s in ("original", "own")]
    refs = [unpack(folder / (name + ".json.gz")) for name in names]
    with (
        worker(
            "long" if source == "anne" else "profile",
            "installed/" + stage + "/" + arm + "/" + source,
        ) as guard,
        TemporaryDirectory() as temp,
    ):
        tmp = Path(temp)
        archive = tmp / "relocated"
        # Preserve the relative object/view relationships in an independent
        # archive tree. APFS clones copy the contents on an unsupported volume.
        subprocess.run(["cp", "-cR", str(ARCHIVE), str(archive)], check=True)
        verify(archive, archive_stage)
        wheel = next((archive / "wheels" / archive_stage).glob("*.whl"))
        policy = archive / "models" / archive_stage / arm
        assert all(
            r["weights"] == read(policy / "registration.json")["weights"] for r in refs
        )
        site = tmp / "installed"
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--offline",
                "--no-deps",
                "--target",
                str(site),
                str(wheel),
            ],
            check=True,
        )
        if source == "anne":
            prompt = tmp / "complete-prompt.txt"
            prompt.write_text(
                Path(task["source"]).read_text() + "\n\n" + task["question"]
            )
            stage_dir = tmp / "staged"
            stage_dir.mkdir()
            opening = [
                "--calculator",
                "--prompt-file",
                str(prompt),
                "--staged-context-dir",
                str(stage_dir),
            ]
        else:
            question, _ = instance(task, "original", tmp)
            provider = (
                ["--calculator"]
                if task["provider"] == "calculator"
                else ["--keyed-csv", task["document"]]
            )
            opening = [*provider, "--prompt", question, "--cache-state-prefill"]
        args = [
            "ask",
            "--checkpoint",
            str(archive / "donor"),
            "--session-policy",
            str(policy),
            "--tool-format",
            "qwen",
            *opening,
            "--follow-up",
            task["followup"],
            "--max-tokens",
            str(cap),
            "--max-turns",
            "3",
            "--no-reuse-prefix",
            "--trace",
        ]
        lease = child_lease()
        lease["env"].update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HOME=str(tmp / "empty-model-cache"),
            LOOPQWEN_COLD_METRICS=str(tmp / "native-metrics.json"),
        )
        tick = monotonic()
        output_path = tmp / "trace.json"
        last_empty = 0.0
        visible_interval = None
        with output_path.open("w") as stdout:
            process = subprocess.Popen(
                [sys.executable, "-I", "-c", GUARD + CODE, str(site), *args],
                cwd=tmp,
                stdout=stdout,
                **lease,
            )
            try:
                while process.poll() is None:
                    guard()
                    if visible_interval is None:
                        observed = monotonic() - tick
                        if output_path.stat().st_size:
                            visible_interval = [last_empty, observed]
                        else:
                            last_empty = observed
                    sleep(0.25)
                if visible_interval is None and output_path.stat().st_size:
                    visible_interval = [last_empty, monotonic() - tick]
                child_metrics = tmp / "native-metrics.json"
                if (
                    child_metrics.exists()
                    and read(child_metrics)["peak_bytes"] >= 8 * 2**30
                ):
                    raise InterruptedError(
                        "Installed MLX process reached the registered peak allocation stop"
                    )
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, args)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        result = json.loads((tmp / "trace.json").read_text())["exchanges"]
        same = [
            trace(r) == trace(ref["trace"]) for r, ref in zip(result, refs, strict=True)
        ]
        callbacks = sum(
            len(t.get("observations", [])) + int("observation" in t)
            for r in result
            for t in r["turns"]
        )
        rec = dict(
            passed=all(same),
            identical=same,
            callbacks=callbacks,
            source=source,
            stage=stage,
            arm=arm,
            cap=cap,
            seconds=monotonic() - tick,
            wheel_sha256=digest(wheel),
            weights=refs[0]["weights"],
            model_cache_empty=True,
            first_stdout_observation_interval=visible_interval,
            child_peak_bytes=max(
                read(child_metrics)["peak_bytes"],
                *(t["peak_bytes"] for r in result for t in r["turns"]),
            ),
            stdout_timing="Cold process start to first observed CLI output byte, including load and both public turns. Model first-visible token timing is reported separately in the development trajectories.",
            network_guard=True,
            results=result,
            compared=[
                str((folder / (name + ".json.gz")).relative_to(ROOT)) for name in names
            ],
        )
        packed(ROOT / "installed" / f"{stage}-{arm}-{source}-{cap}.json.gz", rec)
        if not all(same):
            raise ValueError(
                "Installed replay differs from the paired source trajectory"
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True)
    p.add_argument("--archive-stage", required=True)
    p.add_argument("--arm", choices=("loop", "ordinary"), default="loop")
    p.add_argument("--source", choices=("tiles", "parcels", "anne"), required=True)
    p.add_argument("--cap", type=int, choices=(128, 512, 4096), required=True)
    a = p.parse_args()
    cold(a.stage, a.source, a.cap, a.archive_stage, a.arm)
