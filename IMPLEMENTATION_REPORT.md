# Implementation and validation report

Completed against the repository review and multilingual requirements on 13 November 2026. Usage is documented in [README.md](README.md); the implemented architecture is described in [PLAN.md](PLAN.md).

## Delivered behavior

- Validated domain/language profiles, configurable input mapping, metadata, cleaning, token preservation, routes and long-text chunking.
- Unicode-aware processing, Arabic/English mixed-script detection, configurable Arabic normalization and registered language/tokenizer/discovery extensions. Isolated technical codes such as HTTP no longer trigger a false unsupported language in Arabic input.
- Generic DataFrame/file/API fitting with optional labels; duplicate/conversation grouping and temporal split validation.
- HDBSCAN and adaptive K-Means, selectable frozen PCA/UMAP/no reduction, sampled separation scoring, stability/support/coverage diagnostics, and explicit search budgets.
- Preserved fitted discovery/noise assignments; working single-item transforms and tiny, one-class and all-noise outcomes.
- Class-based TF-IDF, multilingual keywords, optional localized category names and approved route enforcement.
- Classifier suite plus a direct prototype baseline, consistent held-out novelty calibration/evaluation, OOS probe metrics and per-language evaluation slices.
- Domain-isolated immutable taxonomy versions, embedding manifests, stable category IDs and lineage, failed-refit rollback and synchronized model reads.
- Explicit unknown/review response semantics and a persistent masked review queue.

## Verification

| Check | Result |
|---|---|
| `make ci-quality` | Passed: **33 tests**, **93.69% coverage**, Ruff lint and format checks |
| `make ci-docs` | Passed: main and example profile configuration validated |
| `uv build` | Source distribution and wheel built successfully |
| `git diff --check` | Passed |

The optional UMAP path was installed and exercised. Tests include detection/cleaning, cross-language routing with controlled semantic vectors, K beyond the former cap, model reload/isolation, malformed embedding rejection, failed-refit recovery, temporal/grouped splits, API/CLI entry points and review statuses. The existing test fixtures were corrected to avoid duplicate examples across splits and to configure their mock embedding dimensions explicitly.

## Live multilingual integration smoke

The configured gateway was exercised with [24 authored examples](examples/multilingual_smoke.jsonl): 12 English and 12 Arabic narratives spanning payment, delivery and technical incidents. The embedding deployment alias was `rag-embedding`, returning 1,536 dimensions. LLM naming was disabled. Discovery fitted the automatically selected training partition of 16 rows.

| Strategy | Discovered K | Assigned training coverage | New embedding requests for comparison |
|---|---:|---:|---:|
| HDBSCAN | 3 | 68.75% | Cached vectors reused |
| Adaptive K-Means | 3 | 100% | 0 |

These are **integration results on a small authored sample**, not benchmark accuracy or evidence that one strategy is generally better. Some examples are bilingual paraphrases; no accuracy claim is made from these splits. Raw cloud cache/model artifacts remain local under ignored `data/live_smoke/`.

To run the same input through the current defaults:

```sh
DATA_DIR=data/live_smoke ticket-clf --input examples/multilingual_smoke.jsonl \
  --strategy hdbscan_density --no-llm-naming --report-path /tmp/hdbscan-smoke.md
DATA_DIR=data/live_smoke ticket-clf --input examples/multilingual_smoke.jsonl \
  --strategy aligned_kmeans --no-llm-naming --report-path /tmp/kmeans-smoke.md
```

These commands use the configured gateway; its model/configuration affects the result. The live deployment did not declare `EMBEDDING_REVISION`; an immutable revision should be configured before relying on reproducible deployment identity.

## Practical limits

Full deployed-model experiments on Banking77, CLINC/OOS, technical tickets and claims remain empirical validation work. Unit tests and this smoke test do not establish dialect coverage, business taxonomy correctness, lineage accuracy, production latency or cost savings.

The implemented taxonomy is shared across languages. Translation adapters, hierarchical/multi-issue routing and a separate per-language taxonomy strategy remain optional extensions. The configured multilingual representation must be validated for the languages enabled in each profile. Legacy artifacts require refitting, and the old banking benchmark report remains historical evidence.
