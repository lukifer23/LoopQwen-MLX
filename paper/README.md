# Where shared decoder depth earns its cost

The manuscript covers the original shared-span experiments, the matched reasoning controls, the standalone 128K language model, and the addition of executable tool sessions. The paper's numerical results come from native Apple Silicon MLX runs.

- [Read the paper](loopqwen.pdf)
- [Independent clean rebuilds, payload identity and outside-checkout replay](../docs/measurements/reproducible-runtime-wheel.json)
- [Verified isolated wheel replay, including both Hindi failures](../docs/measurements/native-function-wheel.json)
- [Native function conversations, actual actions and both final answers](native-function-requests.csv)
- [Checked calculator on fresh human questions](checked-equations-field.csv)
- [All 256 actual calculator training attempts](calculator-training-attempts.csv)
- [All 120 paired requests after recovery training](calculator-recovery-requests.csv)
- [Re-evaluating the 38 original rejected programs as checked equations](rejected-equations-replay.csv)
- [Original spellings and token sequences for all 5,587 multilingual inputs](localized-inputs.csv)
- [Four process-isolated 50-request streams and tokenizer counts](tokenizer-streams.csv)
- [131K mixed-weight requests, prefill and decode rates, setup costs and physical histories](mixed-weight-requests.csv)
- [Every full 131K request in counterbalanced execution order](full-decoder.csv)
- [Repaired release 04, including all answers, medium replays and numbers of biased native queries](repaired-register-streams.csv)
- [All four compact offline workload streams, with native and process memory](compact-register-streams.csv)
- [Every article and paper response with returned character ranges](document-search-responses.csv)
- [Manuscript source](loopqwen.typ)
- [Both tool families and final-answer accuracy across 78 requests](intent-scores.csv)
- [First executed operation and both operand values against the expert on each financial question](report-first-calculations.csv)
- [Every financial and school-table answer, intermediate call, returned value and prompt length](report-calculation-requests.csv)
- [Executed calculator programs, returned values and final answers on the 24 human questions](intent-arithmetic-actions.csv)
- [The selected action, arguments and actual result for each of 32 live requests](resource-actions.csv)
- [Machine-readable paired scores and timings for the frozen tool model](typed-transfer.csv)
- [Selective refinement scores on all 300 passages by source suite and domain](selective-domains.csv)
- [Fresh source-level loss and accuracy changes](selective-sources.csv)
- [NLL and exact predictions by unseen work in the source-balanced comparison](source-balanced-works.csv)
- [Shared-session lexical overlap for each of the 48 fresh human requests](shared-training-overlap.csv)
- [Overlap between folded-model training contexts and every fresh human prompt](folded-training-overlap.csv)
- [Semantic-judgment input lengths, original post references and lexical overlap](meaning-input-audit.csv)
- [Per-domain next-token scores, costs and document-level uncertainty](raw-language.csv)
- [Training lineage, optimizer exposure and checkpoint hashes](model-lineage.csv)
- [Median and tail tool latency and first-token time](tool-latency.csv)
- [Trajectory-level costs, calls and failure types](tool-trajectories.csv)
- [Training-to-evaluation lexical overlap, with and without normalized numbers](training-overlap.csv)
- [Original reading questions, references and both complete visible answers](passage-answers.csv)
- [Reading results with paired article-level uncertainty](passage-scores.csv)
- [BoolQ sensitivity after excluding near-duplicate training paragraphs](reading-overlap-sensitivity.csv)
- [Full-history answer quality and first-token time per request](register-efficiency.csv)
- [Measured system comparisons against untouched Base](system-comparisons.csv)
- [SQLite plan components, executed denotations and alternate-row tests](sql-action-audit.csv)
- [Counterfactual conversation scores and paired one-pass replay](conversation-replay.csv)
- [Direct page-kernel time, peak allocation and correct outputs](prefill-resource-comparisons.csv)
- [Actual native single-query operator latency and working memory](packed-decode.csv)
- [Prompt and post-first-token throughput with actual denominators](long-throughput.csv)
- [Per-family mixed deployment accuracy, latency, tokens and peak memory](mixed-deployment.csv)
- [Paired sensitivity to leaving out one entire field family](mixed-family-sensitivity.csv)
- [Installed and isolated field CLI calls, observations, answers and timings](field-cli.csv)
- [Correct completed answers at each observed time threshold](correct-by-deadline.csv)
- [Accuracy, first-token latency, total time and memory jointly on all 144 requests](paired-request-tradeoffs.csv)
- [Per-family counts at three measured cost margins](paired-request-tradeoffs-summary.csv)
- [Installed size, transport bytes and restoration time](transport.csv)
- [Release-specific checksums, size reduction, and offline parity](release-transport.csv)
- [Native time charged across research windows](native-compute.csv)
- [Study-level evidence index](study-index.csv)
- [Every registration, threshold, budget and publication status](registry-index.csv)
- [SHA-256 inventories behind the figures](figure-sources.json)

## Rebuild and verify

The commands below rebuild the vector figures, evidence index and PDF. Verification hashes every published payload and registered dataset, checks the recorded user examples against their actual callbacks, and runs each study's archived verifier, including the recovered sessions. For the inventory check alone use `.venv/bin/python scripts/research_registry.py --verify-files`. Older verifiers require the pinned tokenizer in the local Hugging Face cache. To reproduce the runtime distribution separately, use `.venv/bin/python scripts/verify_runtime_wheel.py`. It makes two offline builds from the pinned code revision, compares every file to the delivered wheel, installs the rebuilt wheel into an isolated directory and replays the real file and Hindi package conversations.

```sh
uv sync --locked --group data --group figures --group paper --extra guided
.venv/bin/python scripts/build_paper.py --verify
```

Each row in `study-index.csv` points to a checked-in measurement directory. Its `registration.json` contains the question, budget and stop rule. Its `checksums.json` identifies all recorded files. Some earlier phases use the separate `docs/measurements/` hierarchy and are described in [the phase review](../docs/PROJECT_REVIEW.md).

## Reproduce native tool comparisons

Clone the repository on Apple Silicon and run `.venv/bin/python -m loopqwen download` for the pinned Base. The typed, quantized study also requires the revision-bound official Qwen instruction sibling recorded in its registration. Begin a new bounded research window with `.venv/bin/python scripts/research_session.py`. Run one study at a time. Each root-level entry below references an immutable registration. The runner reads its settings and frozen data. Fresh local outputs go under `outputs/research/`; existing completed outputs are protected against overwrite.

```sh
.venv/bin/python scripts/tool_study.py tool-dialogue-loop-01
.venv/bin/python scripts/tool_study.py tool-dialogue-boundary-01
.venv/bin/python scripts/tool_study.py tool-dialogue-depth-01
.venv/bin/python scripts/tool_bfcl_study.py tool-bfcl-transfer-01
.venv/bin/python scripts/tool_study.py tool-dialogue-grounded-01
.venv/bin/python scripts/tool_study.py tool-dialogue-typed-01
```

The archived comparisons use their checked-in parents. For a read-only check, use `.venv/bin/python scripts/research_archive.py STUDY --verify`. Run `.venv/bin/python scripts/research_registry.py` to verify data, registrations and archive inventory identities. The data builders for the public tasks and training traces pin the Hugging Face revisions. [Data attribution and licenses](../research/licenses/TOOL-DATA-NOTICE.txt)

The calculator recovery study collects real opening failures with the qualified typed adapter before fitting a separate two-pass response to the actual observation. Reproduce its training and both scored arms using `.venv/bin/python -m loopqwen.research.callback_recovery callback-recovery-01`. The frozen training targets and held-out human requests can be rebuilt with `scripts/callback_recovery_material.py`.

## Additional registered comparisons

The list below gives the normal native entry point for each family. Pass the desired immutable study ID as its argument. Before re-running a relative-path example, change to the repository root.

| Runner | Registered workload |
| --- | --- |
| `scripts/contrast_study.py` | Completed free answers, both members of human question edits and an identical-prefix readout |
| `scripts/tool_prefix_study.py` | Balanced repeated measurements with original or cached prefill |
| `scripts/tool_bfcl_study.py` | Independent single- and multi-schema publisher questions |
| `scripts/tool_study.py` | Single-pass tool attribution, instruction-parent control and caller-executed tools |
| `scripts/tool_release.py` | Combined offline build, field callbacks, exact language streams and physical 131K swaps |
| `-m loopqwen.research.balanced_register` | Equal-site-count register graph versus full-history Base |

Use `scripts/research_archive.py STUDY` to seal a completed local run in a new checkout. To verify existing committed evidence use `--verify`. After all measured jobs in a window are archived, run `scripts/research_registry.py --update` and `scripts/research_session.py --close`. All script paths in this table are arguments to the repository's Python interpreter.

Underlying human questions and verified partitions are in `research/data/`. The reproducible material builders pin publisher revisions and SHA-256 hashes. Rebuilding the BoolQ subsets needs PyArrow; numerical inference and stored-result verification do not depend on PyArrow. [BoolQ attribution](../research/licenses/BOOLQ-NOTICE.txt) · [Tool data attribution](../research/licenses/TOOL-DATA-NOTICE.txt)

The field-performance figure compares complete request time and correctness on the same requests. The accompanying CSV reports the number of paired wins and losses and the raw elapsed seconds for both arms. Public function and database tasks require successful execution. HotpotQA uses the reference phrase in a completed answer. The lexical audit covers the original database transfer set, training on separate WikiSQL tables, and public-function retention. [Field performance data](field-performance.csv) · [Training overlap](training-overlap.csv)

| Additional runner | Evidence |
| --- | --- |
| `-m loopqwen.research.bridge_reading` | Multi-hop generation from ten original HotpotQA documents with human-labeled support retrieval |
| `-m loopqwen.research.cache_language` | Causal full-vocabulary predictions from a learned hidden-state cache |
| `-m loopqwen.research.raw_distillation` | Native 2B teacher probabilities and recurrent 0.8B student next-token likelihood |
| `-m loopqwen.research.narrative_book` | Whole-novel question and person-name intervention with physical KV offsets |
| `-m loopqwen.research.register_allocation` | Reserved and incrementally grown full-history storage on identical weights |
| `-m loopqwen.research.folded_system` | One physical gated-delta block at two instruction-model depths, hidden reconstruction, 40 new human questions and 52 historical public/field actions |
| `-m loopqwen.research.report_calculation` | Full issuer-disjoint report and table prompts, alternating real calculator calls, final-number accuracy and numerical reuse |
| `-m loopqwen.research.action_preferences` | External response preferences, targeted file clarification, real CSV reads and public tool retention |
| `-m loopqwen.research.draft_program` | Actual generated reasoning prefixes, native recurrent action training, live calculator observations and fresh human arithmetic |
| `-m loopqwen.research.selective_raw` | Selective larger-teacher supervision, 300 raw passages and incremental cache parity |
| `-m loopqwen.research.linked_reading` | Directed evidence relations and actual ten-document HotpotQA answers |
| `-m loopqwen.research.passage_copy` | Learned vocabulary and copy probabilities on complete human reading passages |
| `-m loopqwen.research.packed_release` | Isolated two-book, short-output and live-callback crossover with the qualified parent package |
| `-m loopqwen.research.full_decode` | Repeated complete-decoder 131K requests, every output token and all retained cache sites |
| `-m loopqwen.research.session_tokenizer` | Four fresh-process streams with real resource calls, switch latency and peak process RSS |
| `-m loopqwen.research.meaning_alignment` | Human same-anchor semantic judgments across Wikipedia and technical support |
| `-m loopqwen.research.transport` | Transport size, restored file inventories and isolated native generation |


The operand, schema and mixed field comparisons use the standard entry points below. Their registrations contain all run caps, sample identities and decision thresholds.

| Runner | Endpoint |
| --- | --- |
| `-m loopqwen.research.operand_study` | Choose arithmetic operations and their arguments with a shared native layer, run a real calculator, then decode the answer |
| `scripts/schema_study.py` | Same-checkpoint comparison of ordinary versus schema-guided generation, with exact prompt and callback replay |
| `-m loopqwen.research.mixed_field` | One product against ordinary Base on 50 interleaved field requests with cold startup and mode switching |

The native compute ledger records charged attempts for each supported closed session. A charge includes the time spent on model work, preparation and scoring under the native worker lease. It is a wall-time resource measure. [Machine time and original ledger hashes](native-compute.csv)

The matched-tool figure reports paired accuracy differences for the two-pass candidate against untouched Base and the equally trained one-pass checkpoint. Each bootstrap sample keeps all questions from a sampled world or public function together and preserves each panel's size. [Plot](matched-tool-effects.svg) · [Points and 95% descriptive intervals](matched-tool-effects.csv)

The timing supplement gives median first-token and complete-session latency, generated tokens and total wall time for every enrolled tool request. It also reports the subset solved by both arms, so readers can inspect speed when both delivered a correct answer. [Paired timing supplement](matched-tool-latency.csv)

Two source supplements track actual follow-up context and actual attention work. [Per-dialogue registered and callback-equivalent outcomes](conversation-replay.csv) · [Full-history causal attention work, offsets and timing](page-prefill.csv)

The recurrent table-parser comparison is run with `python -m loopqwen.research.table_parser_study table-parser-01`. Its material builder re-executes the column- and span-bound plans on the publisher SQLite database before registering them. The primary evaluation also re-executes the inferred plan on a held-out row bootstrap to reduce credit for accidentally matching one result. [Frozen protocol](../research/experiments/table-parser-01.json)

The SQLite action audit compares the model's first proposed operation, column and complete filter set against the publisher query. Text literals are case-folded and numeric literals on real columns are compared as decimals. Grounding uses independent SQLite execution on both the published rows and the frozen row bootstrap. [Per-item action audit](sql-action-audit.csv) · [Figure](sql-action-audit.svg)

The semantic alignment supplement records free responses, common-prefix judgments and answer margins on all 80 requests. A separate audit reports the source text lengths and maximum lexical overlap with fitting material. [Outcomes and times by source](meaning-outcomes.csv) · [Both models on every request](meaning-requests.csv) · [Input audit](meaning-input-audit.csv)

The fresh 131K direct comparison reports unaggregated prompt and output rates, decoded values, timing and physical history receipts. [Per-request supplement](fused-base-requests.csv) · [Side-by-side resource measures](fused-base-full.svg)

The file-dialogue audit compares every enrolled request and follow-up to both the external preference-fitting questions and the original tool-training replay. It reports exact and near matches with and without numerical normalization. [Input overlap](action-preference-inputs.csv)

The arithmetic draft supplement includes the model-generated reasoning length, emitted program, callback result, final number and wall time for both arms on every fresh human question. [Requests and executions](draft-program-requests.csv) · [Human final-number and grounding scores](draft-program.svg)

The single-backbone study reports both user outcomes and hidden reconstruction. [Scores and resources](folded-system.csv) · [All 184 responses](folded-requests.csv) · [Performance by request type](folded-system.svg)

The document-search study uses complete user-supplied text files and preserves character offsets for each returned passage. Reproduce its pinned sources with `.venv/bin/python -m scripts.document_search_material`. The Qasper source archive and original SQuAD development set are downloaded only when absent. [Human-query retrieval reference](document-retrieval.csv) · [Frozen document comparison](../research/experiments/document-search-field-01.json)

Preference training is reported on complete requests, missing-argument conversations and prior public functions. [All 108 model interactions](action-preference-conversations.csv) · [Shared-file cluster counts](action-preference-documents.csv) · [Transfer versus clarification](action-preference.svg)

Mixed-service timing includes every enrolled request, including capped generations. The first request includes cold loading. Compare the request-level first-token and total-time distributions with paired counts, medians and nearest-rank upper quantiles. [Timing curves](service-latency.svg) · [Measured quantiles and paired counts](service-latency.csv)

The earlier selective-language manifest used the final Python filename for all manuals in each split. The corrected chart uses exact publisher paragraph matches across all 544 manual excerpts. Run `.venv/bin/python scripts/selective_source_audit.py --verify` to replay the attribution. [Per-excerpt corrections and original file hashes](../research/data/selective-manual-attribution.json) · [Corrected effects by source](selective-sources.csv)

The source-balanced comparison includes 320 fitting works and 24 primary test works. The catalog lists each Python manual or Wikipedia article and each PG-19 publisher title and publication-year field. [Catalog, work hashes and enrolled windows](source-balanced-catalog.csv)

The intact shared-backbone study exposes its full 120 paired trajectories, physical parameter sharing, actual optimizer target allocation and the separate installed CLI checks. [Per-request outcomes](shared-session-requests.csv) · [Per-family counts and timing](shared-session.csv) · [Loss weight by task](shared-objective-weights.csv) · [Two outside-panel requests](../docs/measurements/shared-session-installed-cli.json)

The trained document-search comparison retains the actual question, answer, call, source ranges, exactness, token F1 and measured cold costs on all 40 external files. The descriptive phrase column checks whether a completed reply contains the normalized source answer. [All 80 responses](document-evidence-responses.csv) · [Original query retrieval](document-evidence-retrieval.csv) · [Results by source](document-evidence.svg)

The workload-mix supplement reweights the observed fresh and retention groups of each shared-backbone comparison. The rates within each group stay fixed. [Counts and break-even fraction](request-mix.csv) · [Accuracy across request mixes](request-mix.svg)

The complete joint-decoder replay used the corrected Qwen3.5 tokenizer and the original trained weights. [Every request and external execution](unified-session-requests.csv) · [Per-family outcomes and costs](unified-session.csv) · [Same 72 requests across both designs](repeated-service-requests.csv) · [Original mismatched Unicode input](../research/results/shared-session-unified-01/tokenizer-parity.json)

The same joint policy was also asked about a broken-jar calculation, the installed NumPy version and the byte count in the real release report. [Complete installed CLI comparisons](../docs/measurements/unified-session-installed-cli.json)

The observation-review experiment separates free exchanges from preferences measured on identical receipts. The source rows include the question, calls, actual returned values, final answer and elapsed time in each arm. The receipt supplement reports both logit margins for each held-out question. [Free exchanges](observation-review-requests.csv) · [Receipt decisions](observation-review-decisions.csv) · [Figure](observation-review.svg)

The document-frequency experiment records the exact evidence interval within each complete Git tree or dependency lockfile, the selected attention interval, free answer and full cold time. [Whole-document measurements](document-frequency-reader.csv) · [Evidence positions and cost](document-frequency-reader.svg)

The question-contrast study tests transfer to an independently authored corpus that supplied no training examples. Both human questions must be answered correctly for a DROP passage to count. [Paired human answers](question-contrast-passages.csv) · [Full requests and callbacks](question-contrast-requests.csv) · [Source-level figure](question-contrast.svg)

DROP passage and question excerpts come from Dua et al., distributed under the publisher’s CC BY-SA 4.0 license. The paired SQuAD material uses the original paragraph, original human questions and numeric labels. The SVAMP material keeps the two human question wordings and records each quantity substitution and executed equation. [DROP paper](https://aclanthology.org/N19-1246/) · [DROP license](https://creativecommons.org/licenses/by-sa/4.0/) · [SQuAD release](https://rajpurkar.github.io/SQuAD-explorer/) · [SVAMP paper](https://aclanthology.org/2021.naacl-main.168/)

The four recent free-generation comparisons also report simultaneous gains in correctness, whole-dialogue time, first-token time and peak MLX allocation. [Every paired cost and score](paired-costs.csv) · [Counts of gains, losses and ties](paired-costs-summary.csv)

The question-contrast supplement includes [independent passage pairs](question-contrast-passages.csv), [all 236 user-request records across the two arms](question-contrast-requests.csv) and [local training overlap by source](question-contrast-overlap.csv). The service audit preserves each [visible answer, requested action and real observation](service-meaning-requests.csv) and the [score under four definitions](service-meaning-summary.csv). The [private-prefix measurements](conversation-document-prefill.csv) record the actual reused length, first-token time, complete time and peak for each source conversation.

The [installed 173,129-token replay](../docs/measurements/document-reader-installed-02.json) reproduces both output streams and all native offsets. The [two-request arithmetic example](../docs/measurements/contrast-reader-installed-02.json) includes the model's reversed subtraction and the incorrect repeated follow-up. Both were run outside checkout from the [same deterministic wheel](../docs/measurements/portable-base-wheel-02.json) without a model cache or network connection.

The complete-book follow-up figure and CSV identify cold opening time, reported answers, real calls and retained histories for each book on each arm. [Per-request data](book-conversation-requests.csv) · [Figure](book-conversation.svg)
