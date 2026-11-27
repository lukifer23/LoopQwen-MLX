"""Verify all saved physical features and reconstruct missing tensors from the pinned native donor."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx
from mlx.utils import tree_flatten

from loopqwen.conditional_reentry import PACKED, from_instruction, resident
from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read
from loopqwen.kernels.gdn_backend import select
from loopqwen.research import conditional_features as original
from loopqwen.research import contrastive_features as subsequent
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.registered import registered
from loopqwen.worker import acquire


def backbone(s, root, restore):
    identity = read(root / "backbone.json")
    dest = Path(s["checkpoint"]).parent / "conditional-reentry-backbone" / PACKED
    if dest.is_file():
        assert (
            dest.stat().st_size == identity["bytes"]
            and digest(dest) == identity["sha256"]
        )
        return dest
    if not restore:
        raise FileNotFoundError(dest)
    acquire()
    net = from_instruction(s)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as tmp:
        packed_path = Path(tmp) / PACKED
        mx.save_safetensors(str(packed_path), dict(tree_flatten(net.text.parameters())))
        assert (
            packed_path.stat().st_size == identity["bytes"]
            and digest(packed_path) == identity["sha256"]
        )
        packed_path.replace(dest)
    return dest


def run(name, restore=False, backbone_only=False, check_rebuild=0):
    s = registered(name)
    root = Path("research/results") / name
    src = read(root / "checksums.json")
    assert digest(root / "registration.json") == src["registration.json"] and s == read(
        root / "registration.json"
    )
    assert digest(root / "backbone.json") == src["backbone.json"]
    path = backbone(s, root, restore)
    if backbone_only:
        print("Verified packed native backbone", path)
        return
    assert digest(root / "feature-index.json.gz") == src["feature-index.json.gz"]
    idx = unpack(root / "feature-index.json.gz")
    where = original.feature_root(s)
    if restore:
        where.mkdir(exist_ok=True, parents=True)
    net = tok = material = None
    repaired = checked = 0
    for i, r in enumerate(idx):
        dest = where / r["file"]
        exists = dest.exists()
        if exists and (
            dest.stat().st_size != r["bytes"] or digest(dest) != r["sha256"]
        ):
            raise ValueError(
                "Existing native file differs from its sealed inventory: " + str(dest)
            )
        if exists and i >= check_rebuild:
            continue
        if not exists and not restore:
            raise FileNotFoundError(dest)
        if net is None:
            acquire()
            net = resident(s["checkpoint"], root, s, read(root / "backbone.json"))
            select(net, "native")
            tok = tokenizer(s)
            assert digest(Path(s["training"])) == s["sources"][s["training"]]
            material = {w["id"]: w for w in unpack(Path(s["training"]))}
        with TemporaryDirectory(dir=where) as tmp:
            target = Path(tmp) / dest.name
            if name == original.NAME:
                measured = original.write_one(tok, net, material[r["id"]], target)
            else:
                measured = subsequent.record(net, tok, material[r["id"]], target)
            assert measured == r, (r["id"], measured["sha256"], r["sha256"])
            checked += 1
            if not exists:
                target.replace(dest)
                repaired += 1
        if (i + 1) % 256 == 0:
            print(i + 1, "native files checked or restored", flush=True)
    print(
        "Verified",
        len(idx),
        "feature files,",
        sum(r["bytes"] for r in idx),
        "bytes,",
        repaired,
        "restored,",
        checked,
        "independently regenerated",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=(original.NAME, *subsequent.FEATURES))
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Recreate missing source-verified tensors on native Metal",
    )
    parser.add_argument("--backbone-only", action="store_true")
    parser.add_argument(
        "--check-rebuild",
        type=int,
        default=0,
        help="Physically regenerate the first N archived feature records in a temporary directory",
    )
    args = parser.parse_args()
    if args.check_rebuild < 0:
        parser.error("--check-rebuild must be nonnegative")
    run(args.name, args.restore, args.backbone_only, args.check_rebuild)
