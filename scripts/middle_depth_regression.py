"""Exercise the installed release and shared native-cache paths under the study lease."""

import argparse
from pathlib import Path
from time import monotonic

import pytest

from loopqwen.io import digest, read, save
from loopqwen.identity import fingerprint
from scripts.middle_depth_budget import ROOT, DevelopmentLimit, worker


def runtime_identity():
    return fingerprint(
        {str(p): digest(p) for p in sorted(Path("loopqwen").rglob("*.py"))}
    )


def run(full=False, module=None):
    files = [
        "tests/test_offloaded_prefill.py",
        "tests/test_reserved_cache.py",
        "tests/test_source_copy.py",
        "tests/test_qwen_tools.py",
        "tests/test_tool_session.py",
        "tests/test_tool_cli_release.py",
    ]
    receipt = "release-regression"
    if full:
        files = ["tests"]
        receipt = "full-regression"
    if module is not None:
        testfile = Path(module)
        if (
            testfile.parent != Path("tests")
            or not testfile.is_file()
            or not testfile.name.startswith("test_")
        ):
            raise ValueError("Expected a repository test module")
        files = [str(testfile)]
        receipt = "isolated-regression/" + testfile.stem
    with worker("profile", receipt) as guard:
        rows = []
        stopped = []

        class Observe:
            def pytest_runtest_setup(self, item):
                try:
                    guard(force=True)
                except DevelopmentLimit as error:
                    stopped.append(error)
                    pytest.exit("Saved native resource stop", returncode=2)

            def pytest_runtest_logreport(self, report):
                if report.when == "call" or report.failed or report.skipped:
                    rows.append(
                        {
                            "test": report.nodeid,
                            "status": report.outcome,
                            "seconds": report.duration,
                            "phase": report.when,
                            **(
                                {"failure": str(report.longrepr)[-2000:]}
                                if report.failed
                                else {}
                            ),
                        }
                    )

        start = monotonic()
        code = pytest.main(["-q", "--disable-warnings", *files], plugins=[Observe()])
        save(
            ROOT / (receipt + ".json"),
            {
                "passed": code == 0,
                "tests": rows,
                "seconds": monotonic() - start,
                "files": files,
                "module_sha": digest(Path(module)) if module else None,
                "runtime": runtime_identity(),
                "stopped": str(stopped[0]) if stopped else None,
            },
        )
        if stopped:
            raise stopped[0]
        if code:
            raise RuntimeError(f"Native release regression exit {code}")


def isolated():
    from scripts.middle_depth_wave import dispatch

    suite = sorted(Path("tests").glob("test_*.py"))
    observations = []
    runtime = runtime_identity()
    for path in suite:
        result = ROOT / "isolated-regression" / (path.stem + ".json")
        done = read(result) if result.exists() else {}
        if (
            not done.get("passed")
            or done.get("module_sha") != digest(path)
            or done.get("runtime") != runtime
        ):
            dispatch("regression", "--module", path)
        observations += read(result)["tests"]
    expected = {r["test"] for r in read(ROOT / "full-regression.json")["tests"]}
    observed = {r["test"] for r in observations}
    if not expected.issubset(observed):
        raise ValueError("The rerun omitted previously supported tests")
    save(
        ROOT / "full-regression-isolated.json",
        {
            "passed": all(r["status"] == "passed" for r in observations),
            "tests": observations,
            "files": len(suite),
            "original": digest(ROOT / "full-regression.json"),
            "additional_tests": sorted(observed - expected),
            "runtime": runtime,
            "method": "One independent native pytest process per module under the same exclusive resource guard.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--module")
    parser.add_argument("--isolated", action="store_true")
    a = parser.parse_args()
    isolated() if a.isolated else run(a.full, a.module)
