"""Exercise the installed release and shared native-cache paths under the study lease."""

from pathlib import Path
from time import monotonic

import pytest

from loopqwen.io import save
from scripts.middle_depth_budget import ROOT, worker


def run():
    files = [
        "tests/test_offloaded_prefill.py",
        "tests/test_reserved_cache.py",
        "tests/test_source_copy.py",
        "tests/test_qwen_tools.py",
        "tests/test_tool_session.py",
        "tests/test_tool_cli_release.py",
    ]
    with worker("profile", "release-and-cache-regression") as guard:
        rows = []

        class Observe:
            def pytest_runtest_setup(self, item):
                guard(force=True)

            def pytest_runtest_logreport(self, report):
                if report.when == "call" or report.failed or report.skipped:
                    rows.append(
                        dict(
                            test=report.nodeid,
                            status=report.outcome,
                            seconds=report.duration,
                        )
                    )

        start = monotonic()
        code = pytest.main(["-q", "--disable-warnings", *files], plugins=[Observe()])
        save(
            ROOT / "release-regression.json",
            dict(
                passed=code == 0, tests=rows, seconds=monotonic() - start, files=files
            ),
        )
        if code:
            raise RuntimeError(f"Native release regression exit {code}")


if __name__ == "__main__":
    run()
