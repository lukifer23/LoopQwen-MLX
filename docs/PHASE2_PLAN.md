# Phase 2 study protocol

Start from d1c6cd60. Preserve the original models, split files and reports. Record the original suite in an isolated output directory. Compare full-context and windowed cached decoding against matching uncached attention on the same token positions. Audit padding, shift, restoration and native derivatives.

Generate 24,000 verified training problems, balanced development and sealed ID/OOD splits. Run a short answer-only and trace pair. Screen curriculum, interpolation, placements and adapter capacity using development examples only. Choose the recurrent candidate by development answer NLL on the common answer-only protocol, report exact answers and retention separately. Keep sequencing, precision, initialization and adapter policy matched within each pair. Use adjusted ranks or a common projection subset for placement screening when the hybrid layer mix differs.

Freeze the winning config and evaluation IDs before confirmatory scoring. Train seeds 43, 53 and 67 from the base. Compare with a dense data-matched twin and a dense control with extra updates calibrated to elapsed training work. Evaluate actual additional sampled candidates under a measured latency budget. Use seed and item pairing, show uncertainty and preserve individual predictions. Run the three-seed local screen before considering substantially longer training.

Profile real Metal forward/backward at 128–512 tokens, measure system memory and test adapter versus fused task outcomes. Commit milestones on main. Phase 2 results use their own namespace.
