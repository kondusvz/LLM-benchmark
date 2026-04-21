# Thesis Patch Notes

This note records benchmark-specific patches applied during the dissertation project so they can be disclosed in the methodology / limitations section.

It is not a full changelog. The aim is to capture the patches that materially affected:

- method faithfulness
- evaluator robustness
- result reporting
- downstream analysis inputs

## Scope

These patches were made to get the benchmark running reliably and to remove avoidable implementation handicaps. They do **not** make every method a strict paper-faithful reproduction. In general, the benchmark should be described as a **benchmark-integrated / deployment-oriented implementation** of the compared methods.

## Method Runner Patches

### `run_openfe.py`

Applied to avoid handicapping OpenFE on mixed/categorical datasets and to fix a multiclass crash:

- switched OpenFE search to use cleaned raw data with categorical columns preserved during search
- sanitized unsafe base feature names before passing them into OpenFE internals
- replaced the brittle helper handoff with the `ofe.transform()` path
- forced `n_data_blocks = 1` for multiclass classification to avoid the LightGBM
  `Number of class for initial score error`
- kept final exported feature matrices numeric-only for evaluator compatibility

Reason:
- the original wrapper path could break or under-represent OpenFE, especially on multiclass and categorical-heavy datasets

### `run_llmfe.py`

Applied to make the benchmarked LLM-FE wrapper closer to the intended method behavior:

- changed the search path to use cleaned raw categorical values instead of pre-encoded numeric-only inputs
- made the generated classification split stratified
- fixed the sample-budget mismatch caused by the upstream sampler's internal `/5` behavior
- removed stale `use_label_encoder=False` noise from the generated XGBoost spec
- kept the final saved feature matrices numeric-only for evaluator compatibility

Reason:
- the earlier wrapper path was too far from the intended search setting and produced avoidable mismatches for categorical datasets and budget control

### `run_llmfe_eff.py`

Added as a separate audit runner for the LLM-FE efficiency investigation.

This path was introduced so the main LLM-FE benchmark results were not overwritten.

Features of the audit path:

- separate output method namespace for the efficiency audit
- richer prompt metadata than the minimal benchmark wrapper
- best-effort paper-inspired candidate evaluation constraints

Important limitation:
- this remains an audit path, not a strict reproduction of the original paper environment

### Upstream LLM-FE repo patches (`repos/LLMFE/...`)

Patches were applied in the local benchmark copy of the upstream LLM-FE repo to support the efficiency audit:

- sandbox/evaluator changes so candidate evaluation timeout handling actually worked in the executed path
- best-effort memory-cap enforcement
- Windows multiprocessing/result-handoff fixes so child-process evaluation returned stable scalar scores

Reason:
- the public repo/config exposed paper-style timeout settings, but the executed sandbox path did not cleanly enforce them in the benchmark environment

### `run_featllm.py` and `repos/FeatLLM/utils.py`

Applied to make FeatLLM work on the `nursery` multiclass dataset:

- patched multiclass AUC evaluation so the full class list is passed explicitly
- replaced the fragile multiclass AUC path with a robust one-vs-rest average over only evaluable classes
- skipped degenerate classes in a given split instead of crashing when a very rare class was absent from `y_test`

Reason:
- `nursery` contains an extremely rare class (`recommend`, 2 rows total), which caused sklearn multiclass ROC AUC to fail on some splits

### `run_ownm.py`

- removed stale `use_label_encoder=False` from the active XGBoost path

Reason:
- the argument was unused and only generated warning noise in the benchmark logs

## Evaluator / Reporting Patches

### `evaluator.py`

Applied to improve robustness and report quality:

- cleaned and reformatted the TXT benchmark report to be more publication-friendly
- added clearer method display names
- improved table formatting and summary sections
- patched nursery-related multiclass AUC handling in the benchmark evaluation flow so missing test-fold classes no longer made AUC unusable at report level

Reason:
- the original report format was rough and some multiclass edge cases produced misleading `N/A` values

## Stability / DKU Analysis Patches

### `random/analyze_feature_stability.py`

- added `--include-methods`
- added `--exclude-methods`
- added repo-root path handling so the script can be run from subdirectories

Reason:
- final thesis stability outputs needed to be generated from the exact in-scope method roster, not from whatever happened to exist in `features/`

### `tools/score_domain_knowledge_utilisation.py`

Applied to make DKU scoring reliable enough for descriptive dataset characterization:

- added `--summary-csv` and `--run-csv`
- added `--mode per-method|pooled`
- added `--dataset` for targeted reruns
- added `--pool-exclude-methods` so pooled DKU excludes methods with anonymous feature names
- routed both pooled and per-method classification through batched classification
- added batching controls (`--batch-size`, pooled feature caps)
- changed missing-classification handling from hard failure to soft failure when coverage is at least 80%
- added `n_unclassified`
- added `n_named_features`, `pct_named_features`, `n_classifiable_features`
- added low-confidence warnings for low named-feature coverage
- excluded anonymous names like `autoFE_f_*` and `feat_*` from pooled classification input
- added `confidence_note` and `dku_band`
- added enriched DKU summary output joined to dataset-registry-style fields
- manually resolved the `HousingData` pooled DKU edge case
- manually reran / merged `blood` in the cleaned pooled outputs

Reason:
- naive pooled DKU was badly distorted by anonymous feature names from classical / rule-based pipelines and by oversized prompt payloads

## MATLAB / Figure Data Tooling

### `tools/load_thesis_data_matlab.m`

Added and iteratively cleaned up as the main MATLAB workspace loader for figure generation:

- consolidated benchmark CSV, DKU summary, and meta JSON loading
- added token-attribution backfilling into the same run-level metadata table
- simplified top-level workspace variables to the tables actually needed for figures
- standardized dataset registry naming and merged DKU fields

Reason:
- the original figure workflow was too fragmented and difficult to use interactively in MATLAB

## Token Attribution

### `tools/attribute_token_usage.py`

Used to backfill missing token usage for methods that were run without native token logging, especially CAAFE.

Important usage note:

- token backfill is attribution-based rather than natively logged
- it should therefore be described as an inferred lower-bound / matched-request estimate, not as first-party method telemetry

## Recommended Thesis Disclosure

Suggested wording:

> Several benchmark-integrated patches were required to make the compared methods runnable and comparable in a shared evaluation environment. These patches primarily addressed data-format compatibility, multiclass evaluation edge cases, logging/reporting quality, and token attribution. Where such changes materially affected method behavior, results are reported as benchmark-integrated implementations rather than strict reproductions of the original papers.

For LLM-FE specifically, an additional note is appropriate:

> An auxiliary efficiency-audit variant was also implemented to investigate the effect of paper-inspired evaluation constraints. This variant should be interpreted as a targeted audit configuration rather than part of the main benchmark comparison.
