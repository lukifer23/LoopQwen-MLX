import json
from pathlib import Path
import struct

import numpy as np
import pytest

from loopqwen.io import digest, save
from loopqwen.weight_transport import BASE, TOOL, INVENTORY, pack, unpack


def write_tensors(path, arrays):
    header = {}
    payload = bytearray()
    for name, dtype, data in arrays:
        end = len(payload) + data.nbytes
        header[name] = dict(
            dtype=dtype, shape=list(data.shape), data_offsets=[len(payload), end]
        )
        payload.extend(data.tobytes())
    encoded = json.dumps(header).encode()
    p = Path(path)
    p.parent.mkdir(exist_ok=True, parents=True)
    p.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def test_quantized_residuals_and_nonquantized_tensors_roundtrip(tmp_path):
    rng = np.random.default_rng(481)
    src = tmp_path / "src"
    raw = rng.uniform(-1, 1, (9, 128)).astype("<f4")
    base = (raw.view("<u4") >> 16).astype("<u2")
    sc = np.full((9, 2), 15360, dtype="<u2")
    bias = np.full_like(sc, 49024)
    packed = rng.integers(0, 2**32, (9, 32), dtype="<u4")
    write_tensors(
        src / BASE, [("a.weight", "BF16", base), ("b.weight", "BF16", base[:2])]
    )
    write_tensors(
        src / TOOL,
        [
            ("a.weight", "U32", packed),
            ("a.scales", "BF16", sc),
            ("a.biases", "BF16", bias),
            ("b.weight", "BF16", base[:2] + 1),
        ],
    )
    (src / "LICENSE").write_text("unit fixture")
    save(src / INVENTORY, {f: digest(src / f) for f in (BASE, TOOL, "LICENSE")})
    arc = tmp_path / "combined.lqzip"
    target = tmp_path / "restored"
    pack(src, arc)
    unpack(arc, target)
    assert {
        p.relative_to(target): digest(p) for p in target.rglob("*") if p.is_file()
    } == {p.relative_to(src): digest(p) for p in src.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        unpack(arc, target)
