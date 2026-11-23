import platform
import re
import subprocess
from importlib.metadata import version
from pathlib import Path

import psutil


def doctor():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Use native arm64 Python on a Metal-capable Mac.")
    import mlx.core as mx
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("MLX Metal GPU is required.")
    gpu = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], text=True)
    return dict(device=mx.device_info(), default_device=str(mx.default_device()),
                macos=platform.mac_ver()[0], python=platform.python_version(), arch=platform.machine(),
                cpu_cores=psutil.cpu_count(), gpu_cores=int(re.search(r"Total Number of Cores: (\d+)", gpu)[1]),
                available_bytes=psutil.virtual_memory().available, metal_test=(mx.ones(3) @ mx.ones(3)).item(),
                versions={m: version(m) for m in ("mlx", "mlx-lm", "mlx-metal", "transformers")},
                kernels={k: hasattr(mx.fast,k) for k in ("gated_delta_update","cross_entropy","rms_norm","scaled_dot_product_attention","metal_kernel")},
                transforms={k:hasattr(mx,k) for k in ("compile","checkpoint","value_and_grad")},
                expected_bf16_text_bytes=1_505_000_000,
                checkpoints=sorted(str(p.parent) for p in Path("checkpoints").glob("*/latest.json")))
