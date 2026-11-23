# Local results on Apple M3 Pro

## 200-update paired pilot

Training adapted the loop. Enabling the six-layer recurrence without training cut procedural exact accuracy from 28/110 to 14/110. Training the shared adapters brought it to 50/110. The equally trained dense twin reached 61/110. Among their paired answers, both models got 47 right, the dense twin alone got 14 right, and the loop alone got three right. The trained loop's answer NLL was 0.6392 versus 0.5613 for the dense twin. Its prose NLL was also higher. This run favors the dense control.

Each twin processed 1,600 examples, 41,833 nonpadding input tokens and 8,540 supervised next-token labels. The sequence cap was 64, with eight accumulated single-document microbatches. Average completed update time was 0.654 s dense and 0.870 s looped. Total training-loop wall time, including validation and checkpointing, was 135.0 s and 179.2 s respectively. The effective layer-execution multiplier was 1.25 and the measured update-time multiplier was 1.33. Both peaked near 1.83 GiB allocated by MLX.

Applying the trained loop checkpoint at K=1,2,3 yielded 51, 50 and 43 correct answers. K=2 had the lowest answer NLL of those three depths. An increased dense output-token allowance left its procedural accuracy and actual work unchanged because it already stopped at EOS. All primary arms scored 0/16 on short-answer GSM8K and 0/8 on the selected numeric MATH slice. These numbers describe this prompt and decoding protocol. [Full pilot comparison](PILOT.md) · [Machine-readable results](measurements/pilot_comparison.json)

## 1,000-update paired experiment

Each extended twin processed the same 4,000 examples, 103,637 input tokens and 21,289 supervised labels. Dense used 0.233 s per optimizer update, 444.5 input tokens/s and 2.70 GiB peak allocation. The recurrent model used 0.329 s, 315.4 input tokens/s and 2.69 GiB. Total training-loop wall times were 239.1 and 336.3 seconds. The final training batch losses were 0.7357 and 0.8299. The final fixed validation slice gave answer NLLs of 0.2216 and 0.2679.

The November 23, 2026 pilot checkpoints record training commit `dc18b825073cbb00374305b044bd5aa247dad741`. Both extended runs record `eb7569d61f64523c15802fe27ce6ffc3fa91b86f`. Each pair used seed 41. All checkpoint generations retain their fully resolved configuration and data-order digest.

## Hardware, precision and performance

The host is an Apple M3 Pro with 5 performance and 6 efficiency CPU cores, 14 GPU cores and 18 GiB unified memory. At setup, 5.23 GiB was reported available. The environment used macOS 27.3, arm64 Python 3.12.12, MLX 0.32.4 and MLX-LM 0.32.2. The text state contains 752,390,432 BF16 and 2,592 FP32 parameters. The K=2 wrapper references exactly the same 320 tensors as the baseline. Both have 752,393,024 unique parameters before adapting. The 46 LoRA modules in the six-layer block add 1,336,320 FP32 parameters, for a total of 753,729,344. The merged model returns to 752,393,024 parameters.

At 64 prompt tokens the unadapted K=1,2,3 models decoded at 69.1, 57.2 and 47.0 tokens/s. The full K=2 checkpointed value-and-gradient pass on one 64-input-token document took 146.3 ms with the native GDN path and 545.5 ms with the upstream reference. Peak allocation fell from 2.95 to 1.82 GiB. Disabling activation checkpointing brought native time down to 129.1 ms at 2.10 GiB. Projecting the four scored rows of this 64-position document plus fused cross-entropy took 8.57 ms for loss and hidden gradient versus 19.69 ms with full projection and standard cross-entropy. Relative hidden-gradient error was zero on the measured document. [Benchmark table](BENCHMARKS.md) · [Raw benchmark and calibration data](measurements/benchmark.json)

The initial isolated 64-token GDN value-and-gradient measurement was 2.81 ms native versus 24.10 ms reference. The final benchmark measured 3.05 ms versus 22.99 ms. The largest relative gradient error across the isolated 32- and 64-token model-shaped initial cases was 0.00292. The largest pretrained GDN projection-parameter gradient error was 0.00280. These are relative L2 errors. Native backward reduction order introduces small run-to-run variation. [Initial primitive measurements](measurements/gdn.json) · [Projection gradients](measurements/gdn_projection.json) · [Repeatability diagnostic](measurements/backward_repeat.json)

## Verification and interpretation

K=1 logits matched the stock language model exactly. Cached and uncached decoding agreed on every greedy token of the short and 49-token test sequences at all three depths. The checkpoint round-trip test recovered every saved trainable tensor and optimizer tensor exactly. The BF16 merge changed verification-prompt logits by 1.62% relative RMS. Reloading the merged model with its bundled runtime reproduced the entire merged logit tensor byte-for-byte in a fresh process. [Architecture probe](measurements/probe.json) · [Cache checks](measurements/cache.json) · [Merged export verification](measurements/pilot_export.json)

The validation placement sweep used 16 calibration problems and frozen pretrained weights. Answer NLLs for six-layer spans starting at 6, 8, 10, 12 and 16 were 1.129, 0.969, 1.574, 1.504 and 0.982. These measurements describe untrained perturbations. All matched runs retained the preselected 10–15 span.

The evaluation uses one seed and a small set of familiar task families. Arithmetic answer training and EOS supervision are only a small part of the 100M-token full-weight reasoning-trace experiment in the literature. Pretraining exposure to public benchmarks is unknown. The six prose documents give a limited retention check. The strict first-line numeric scorer can reject a correct answer embedded in an explanation. Full fine-tuning, quantized export, adaptive-depth training and large reasoning-trace training were outside the measured runs.
