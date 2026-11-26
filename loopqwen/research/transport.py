"""Measure lossless paired-backbone transport and verify a restored offline model."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED

from ..evaluation.mechanism import packed, unpack as records
from ..io import digest, read, save
from ..weight_transport import BASE, INVENTORY, pack, unpack
from .core import Budget
from .registered import registered


def outcome(spec, report):
    return dict(
        passed=report["bytes"] <= spec["raw_ratio"] * report["raw_bytes"]
        and report["bytes"] <= spec["deflate_ratio"] * report["ordinary_zip_bytes"]
        and report["identical"]
        and report["offline_parity"],
        **report,
    )


def verify(root, decision):
    spec = read(root / "registration.json")
    r = read(root / "report.json")
    prior = Path(spec["requires"])
    expected = read(prior / "export-files.json")
    expected[INVENTORY] = digest(prior / "export-files.json")
    # The release inventory hashes all payloads and is itself restored byte for byte.
    restored = read(root / "restored.json")
    orig = read(root / "original.json")
    assert restored == orig and {k: h for k, h in orig.items() if k != INVENTORY} == {
        k: h for k, h in expected.items() if k != INVENTORY
    }
    from scripts.tool_release import tokens

    assert r["identical"] == (restored == orig) and r["offline_parity"] == (
        tokens(records(root / "portable.json.gz"))
        == tokens(records(prior / "portable.json.gz"))
    )
    assert (
        r["raw_bytes"] == sum(read(root / "sizes.json").values())
        and outcome(spec, r) == decision
    )


def run(name):
    spec = registered(name)
    src = Path(spec["checkpoint"])
    prior = Path(spec["requires"])
    out = Path("outputs/research") / name
    if (out / "decision.json").is_file():
        raise FileExistsError(out)
    if (
        digest(prior / "checksums.json") != spec["prerequisite_sha256"]
        or not read(prior / "decision.json")["passed"]
    ):
        raise ValueError("Unqualified transport source")
    budget = Budget(name, spec["limits"])
    target = Path(spec["destination"])
    out.mkdir(parents=True, exist_ok=True)
    try:
        original = read(src / INVENTORY)
        original[INVENTORY] = digest(src / INVENTORY)
        sizes = {k: (src / k).stat().st_size for k in original}
        with TemporaryDirectory(dir=src.parent) as tmp:
            tmp = Path(tmp)
            standard = tmp / "standard.zip"
            at = perf_counter()
            with ZipFile(
                standard, "x", compression=ZIP_DEFLATED, compresslevel=6
            ) as archive:
                for k in sorted(original):
                    archive.write(src / k, k)
                    budget.check()
            ref_seconds = perf_counter() - at
            ref_bytes = standard.stat().st_size
            at = perf_counter()
            packed_path = tmp / "paired.zip"
            pack(src, packed_path)
            pack_time = perf_counter() - at
            budget.check()
            at = perf_counter()
            restored = tmp / "restore"
            unpack(packed_path, restored)
            unpack_time = perf_counter() - at
            budget.check()
            checks = {k: digest(restored / k) for k in sizes}
            from scripts.tool_release import portable, tokens

            test = portable(restored, "mlx")
            expected = records(prior / "portable.json.gz")
            packed(out / "portable.json.gz", test)
            report = dict(
                bytes=packed_path.stat().st_size,
                sha256=digest(packed_path),
                raw_bytes=sum(sizes.values()),
                ordinary_zip_bytes=ref_bytes,
                raw_base_weight_bytes=sizes[BASE],
                ordinary_zip_seconds=ref_seconds,
                pack_seconds=pack_time,
                restore_seconds=unpack_time,
                identical=checks == original,
                offline_parity=tokens(test) == tokens(expected),
            )
            decision = outcome(spec, report)
            save(out / "original.json", original)
            save(out / "restored.json", checks)
            save(out / "sizes.json", sizes)
            save(out / "report.json", report)
            if decision["passed"]:
                if target.exists():
                    raise FileExistsError(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                packed_path.rename(target)
            save(out / "decision.json", decision)
            print(decision, flush=True)
    finally:
        budget.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    run(p.parse_args().name)
