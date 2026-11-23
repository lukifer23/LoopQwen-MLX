# Troubleshooting

- Run `scripts/bootstrap.sh` from a native Terminal session. Doctor checks Metal by evaluating a device dot product. A Rosetta Python or CPU default device fails the backend check.
- Run `python -m loopqwen download` to populate the Hugging Face cache with the pinned Base revision. The chat checkpoint has a different identity. For an offline run, prepare the snapshot and build the data once while connected.
- Use `python -m loopqwen data --procedural-only` if a public data source is unavailable. A previous run with another manifest requires a distinct name. To reproduce the published pilot use the default public mixture.
- Use `python -m loopqwen cache-check` for generation disagreement. Give each prompt a fresh cache and keep its depth fixed. Quantization, changing precision and using a sliding context change the comparison.
- The upstream short-query attention Metal VJP can raise a missing log-sum-exp error. The training collator pads to at least 16 positions and masks loss on the extra tokens.
- If unified memory is scarce, close the competing workload or lower the sequence cap and batch size in a new run. The trainer's minimum settings stop under continued pressure. Compare only twins that pass the manifest and order checks.
- The reference GDN backend is available using `gdn_backend: reference`. Run `gdn-check` after changing MLX or MLX-LM. Benchmarks on another chip, OS or library version can differ.
- A missing checkpoint means its training command has to finish an update at a save boundary. The original config resumes a checkpoint automatically. The export command writes to a new directory to preserve prior exports.
