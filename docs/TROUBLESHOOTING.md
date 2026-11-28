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

On the Phase 5 host, the six historical merged exports totaled 8.5 GiB before compaction. Verified APFS repacking shared 4.769 GiB of equal content across four weight files and their tokenizer assets. Volume free space rose from about 147 to 152 GiB during the operation. Checkpoint generations, optimizer moments, split manifests and Git artifacts stayed at their original paths. The concurrent training run makes the change in volume free space an approximate observation.

Phase 6 uses one checkout supervisor and the user-wide Metal lease. Use `python -m loopqwen resources` to inspect ownership. A memory interruption saves an optimizer checkpoint at the logical update boundary. Before resuming, check available system memory and macOS pressure. The recorded execution policy caps MLX peak at 6 GiB.

For Phase 7, an MLX peak below the per-process limit can coincide with system-wide memory pressure. Check `sysctl -n kern.memorystatus_vm_pressure_level` and `python -m loopqwen resources`. A saved safety endpoint can still be evaluated and verified. The trained weights and the optimizer generation have separate digests. The compact inference archive contains only the trainable tensors and recovery metadata.

When Base appears to have answered but the score is malformed, inspect the generated token IDs through the first `<|im_end|>`, the closing thinking marker and the visible numeric field. The registered Base calibration preserved both raw generations and a corrected first-turn rescore. [Calibration audit](PHASE7_RESULTS.md#thinking-enabled-language-calibration)

For a research release marked as a suffix composition, load the frozen parent and suffix component with `scripts/research_predict.py`. For recovery, the local optimizer and adapter generation in `model/recovery.json` must match the published hashes. An expired research session requires a new explicitly authorized execution window. Earlier archive verification and inference remain available independently of training-session limits.

## Memory monitor is waiting while macOS reports normal pressure

Check available unified memory as well as OS pressure. In the interrupted interleaving study, the 6.4 GiB launch margin applied before model load. During training the registered guards check OS warning or critical pressure, available memory below 1.5 GiB, and MLX peak allocation above 5 GiB. These quantities measure different things. The resumed monitor observed 288 normal-pressure readings and a maximum of 5.98 GiB available. The owner then stopped the monitor and closed the window. A lock file may remain on disk after exit. Check its live holder with `lsof` and verify the optimizer cursor before a future authorized run. [Recovery guide](NEXT_EXPERIMENTS.md)

## Storage in the middle-span study

The full source-supervised pair shares 1,527,751,054 bytes of donor files. The 288 frozen-prefix training features occupy 287,093,261 bytes and serve both arms. The three retained full-document prefix manifests and registered features accounted for 5,194,682,728 bytes of unique working files at the source-supervised cut. The registered limit for rebuildable working data is 16 GiB, with a 24 GiB free-space floor. The cache manager reclaims verified derived prefixes when more space is needed, recording their source, reconstruction command and file hashes. Saved optimizer generations and published model archives have separate lifetimes. [Storage inventory](measurements/middle-depth/retention/source-supervised.json) · [Verified shared book states](measurements/middle-depth/prefix-audits/epoch-one-anne.json) · [Checkpoint archive](measurements/middle-depth/archive-source-supervised.json)

The legacy indexed-feature audit read four saved partial indexes. It identified 25,344 feature paths and verified 138 equal payloads totaling 53,414,636 bytes before repacking them as independently writable APFS clones. Every original loader path still resolves. This count measures verified equal bytes. The volume’s physical free-space change also depends on pre-existing extent sharing, concurrent activity and snapshots. Run `.venv/bin/python -m scripts.compact_exports --indexed-features` to audit the same paths. [Paths, hashes and byte counts](measurements/indexed-feature-storage.json)

For the largest legacy feature directory, eight evenly spaced indexed shards totaled 12,537,980 bytes. Single-threaded zstd level 3 reduced them to 11,675,912 bytes (6.9%). All eight restored SHA-256 hashes matched their index. macOS `ditto --hfsCompression` left allocated blocks unchanged on this sample. We kept the working shards in their directly loadable form. These figures describe the sample only. [Compression measurements and source hashes](measurements/middle-depth/retention/lossless-feature-sample.json)

The packed 04 and 05 transport archives share 1,512,833,024 bytes of aligned equal blocks through APFS copy-on-write. Both original filenames and separate release hashes verified after repacking. Volume free space rose by 1,505,300,480 bytes during that operation with the native study running. [Historical release hashes and allocation observation](measurements/middle-depth/retention/transport-archive-clone.json)
