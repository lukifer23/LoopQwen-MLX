# Trained adapters

Each run contains approximately 5.36 MB of FP32 LoRA weights and its original checkpoint metadata, including all training steps, data manifest, base revision, adapter policy and recurrent schedule. All four models use the same official Qwen3.5-0.8B-Base weights. The underlying base weights are downloaded separately. The files here support inference and merged export. The locally trained resumable generations also include AdamW state in `checkpoints/`.

From a fresh clone and the activated project environment, run

```sh
python -m loopqwen download
python -m loopqwen generate --checkpoint adapters/extended_loop --prompt 'Question: 17 × 23
Answer:' --max-tokens 16
```
