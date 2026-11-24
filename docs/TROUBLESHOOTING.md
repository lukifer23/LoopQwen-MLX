# Troubleshooting

- Run `scripts/bootstrap.sh` from a native Terminal session. Doctor checks Metal by evaluating a device dot product. A Rosetta Python or CPU default device fails the backend check.
- Run `python -m loopqwen download` to populate the Hugging Face cache with the pinned Base revision. The chat checkpoint has a different identity. For an offline run, prepare the snapshot and build the data once while connected.
- Use `python -m loopqwen data --procedural-only` if a public data source is unavailable. A previous run with another manifest requires a distinct name. To reproduce the published pilot use the default public mixture.
- Use `python -m loopqwen cache-check` for generation disagreement. Give each prompt a fresh cache and keep its depth fixed. Quantization, changing precision and using a sliding context change the comparison.
- The upstream short-query attention Metal VJP can raise a missing log-sum-exp error. The training collator pads to at least 16 positions and masks loss on the extra tokens.
- If unified memory is scarce, close the competing workload or lower the sequence cap and batch size in a new run. The trainer's minimum settings stop under continued pressure. Compare only twins that pass the manifest and order checks.
- The reference GDN backend is available using `gdn_backend: reference`. Run `gdn-check` after changing MLX or MLX-LM. Benchmarks on another chip, OS or library version can differ.
- A missing checkpoint means its training command has to finish an update at a save boundary. The original config resumes a checkpoint automatically. The export command writes to a new directory to preserve prior exports.

## Phase 3

Use `python -m loopqwen resources` to see the owner of the user-wide GPU lease and processes in the current checkout. Workers release the kernel lock when they exit. Removing a live lock file can create a second inode and defeat exclusion. A normal controlled interruption finishes the active update and writes an atomic checkpoint. Restart the same stage after the worker exits.

A strict resume checks the training source files, dependency versions, data manifest, optimizer schema, configuration and sample schedule. Restore the recorded checkout and environment for exact continuation. To deliberately change implementation, fork to a new run with an explicit migration reason. An extension preserves AdamW state. A weights-only fork starts its optimizer and cursor at zero. The parent generation remains available.

## Phase 4

Run `python -m loopqwen resources` to inspect the per-user GPU lease and this checkout's process tree. A completed arm has both a completed run journal in `outputs/phase4/runs/` and a matching immutable checkpoint generation. Reissue the same study stage to resume an interrupted arm. Strict training identity includes objective and batching code. For historical exact continuations, use the recorded source revision. An explicitly documented compatible fork can transfer overlapping adapter tensors to a wider adaptation policy with a fresh optimizer.

If a cap cannot fit a single document, raise the cap or register a new sequence-length cohort. Swapping or checkpointing changes belong in the recorded execution policy. Try a lower cap in a measured profile first. Scientific controls require the same logical examples and boundaries across paired arms.

## Disk use on an APFS Mac

Use `du -sh exports checkpoints outputs adapters .git` to locate large directories and `df -h .` for volume free space. Each full BF16 export occupies roughly 1.4 GiB of logical file space. APFS can share many extents across independently writable exports. Logical sizes remain the same after compaction.

With finished exports idle, run `.venv/bin/python scripts/compact_exports.py` for the read-only comparison. Add `--apply` to clone equal-size files, copy differing blocks and SHA-256 verify the complete replacement before an atomic rename. The tool only considers weights and tokenizer assets in local `exports/`. Existing shared blocks and snapshots affect the observed change in free space. Keep training checkpoints, milestone pointers and compact published adapters for recovery and reproduction. Use `python -m loopqwen resources` to inspect the current GPU owner and checkout workers.
