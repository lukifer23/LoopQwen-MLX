import platform
import re
import subprocess
from importlib.metadata import version
from pathlib import Path

import psutil


def require_metal():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Use native arm64 Python on a Metal-capable Mac.")
    import mlx.core as mx

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("MLX Metal GPU is required.")


def doctor():
    import mlx.core as mx

    require_metal()
    gpu = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], text=True)
    core_match = re.search(r"Total Number of Cores: (\d+)", gpu)
    return dict(
        device=mx.device_info(),
        default_device=str(mx.default_device()),
        metal_available=mx.metal.is_available(),
        hardware_model=subprocess.check_output(
            ["sysctl", "-n", "hw.model"], text=True
        ).strip(),
        macos=platform.mac_ver()[0],
        python=platform.python_version(),
        arch=platform.machine(),
        cpu_cores=psutil.cpu_count(),
        gpu_cores=int(core_match[1]) if core_match else None,
        cpu_clusters=[
            int(x)
            for x in subprocess.check_output(
                [
                    "sysctl",
                    "-n",
                    "hw.perflevel0.physicalcpu",
                    "hw.perflevel1.physicalcpu",
                ],
                text=True,
            ).split()
        ],
        available_bytes=psutil.virtual_memory().available,
        metal_test=(mx.ones(3) @ mx.ones(3)).item(),
        versions={
            m: version(m) for m in ("mlx", "mlx-lm", "mlx-metal", "transformers")
        },
        kernels={
            k: hasattr(mx.fast, k)
            for k in (
                "gated_delta_update",
                "cross_entropy",
                "rms_norm",
                "scaled_dot_product_attention",
                "metal_kernel",
            )
        },
        transforms={
            k: hasattr(mx, k) for k in ("compile", "checkpoint", "value_and_grad")
        },
        expected_bf16_text_bytes=1_505_000_000,
        checkpoints=sorted(
            str(p.parent) for p in Path("checkpoints").glob("*/latest.json")
        ),
    )


def gpu_utilization():
    """Latest driver-wide occupancy sample, available without elevated privileges."""
    raw = subprocess.run(
        ["ioreg", "-r", "-c", "AGXAccelerator", "-l", "-w", "0"], capture_output=True
    )
    match = re.search(rb'"Device Utilization %"\s*=\s*(\d+)', raw.stdout)
    return int(match[1]) if raw.returncode == 0 and match else None
