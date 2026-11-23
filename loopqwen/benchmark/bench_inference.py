from dataclasses import replace
from pathlib import Path
from statistics import median
import mlx.core as mx
import yaml
from .bench_gdn import bench_gdn, check_projection_gradients, timed
from .bench_training import bench_training
from ..config import config
from ..checkpoint import setup
from ..data.build_data import load_split
from ..evaluation.generation_eval import generate
from ..training.trainer import evaluate_rows
from ..hardware import doctor
from ..io import save


def benchmark(path):
    settings = yaml.safe_load(Path(path).read_text())
    c = config(settings["training_config"])
    n = settings["repetitions"]
    # Inference and placement use the untouched pretrained projections.
    from ..looped_model import LoopedModel
    from ..model_loader import load_base

    base, tok, info = load_base()
    model = LoopedModel(base)
    model.eval()
    rows = load_split("validation")[: settings["placement_validation_count"]]
    x = mx.array(
        [
            tok.encode(
                "One two three four five six seven eight. "
                * settings["sequence_length"]
            )[: settings["sequence_length"]]
        ]
    )
    prompt = tok.decode(x[0].tolist())
    inference = []
    for k in settings["depths"]:
        model.span = replace(model.span, passes=k)
        generate(model, tok, prompt, 2)
        mx.clear_cache()
        mx.reset_peak_memory()
        samples = [
            generate(model, tok, prompt, settings["decode_tokens"]) for _ in range(n)
        ]
        inference.append(
            dict(
                k=k,
                **{
                    key: median(r[key] for r in samples)
                    for key in ("ttft_s", "prefill_tps", "decode_tps", "peak_bytes")
                },
                per_decode_ms=1000 / median(r["decode_tps"] for r in samples),
                prompt_tokens=samples[0]["prompt_tokens"],
            )
        )
    placements = []
    for start in settings["placement_starts"]:
        model.span = replace(model.span, start=start, passes=2)
        mx.clear_cache()
        mx.reset_peak_memory()
        ms = timed(lambda m=model: m(x), n)
        placements.append(
            dict(
                start=start,
                full_forward_ms=ms,
                validation=evaluate_rows(model, rows),
                peak_bytes=mx.get_peak_memory(),
            )
        )
    del base, model
    mx.clear_cache()
    model, _, _ = setup(c)
    # Profile a complete real document near the requested sequence length.
    profile_row = min(
        load_split("train"),
        key=lambda r: abs(len(r["ids"]) - 1 - settings["sequence_length"]),
    )
    training = bench_training(model, [profile_row], c, n)
    del model
    mx.clear_cache()
    result = dict(
        hardware=doctor(),
        model=info,
        config=c,
        settings=settings,
        inference=inference,
        placements=placements,
        training=training,
        gdn=bench_gdn(),
        projection=check_projection_gradients(),
    )
    save("outputs/benchmark.json", result)
    lines = [
        "# Metal benchmark",
        "",
        "| K | Prefill tok/s | Decode tok/s | TTFT ms | Peak MLX GiB |",
        "|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['k']} | {r['prefill_tps']:.1f} | {r['decode_tps']:.1f} | {1000 * r['ttft_s']:.1f} | {r['peak_bytes'] / 2**30:.2f} |"
        for r in inference
    ]
    lines += [
        "",
        "| GDN training backend | Checkpoint layers | Forward ms | Value + gradient ms | Peak MLX GiB |",
        "|---|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r['backend']} | {r['checkpointing']} | {r['forward_ms']:.1f} | {r['combined_ms']:.1f} | {r['peak_bytes'] / 2**30:.2f} |"
        for r in training["rows"]
    ]
    Path("outputs/benchmark.md").write_text("\n".join(lines) + "\n")
    return result
