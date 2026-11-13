# Multilingual ticket and claim discovery

A configurable pipeline for cleaning text, embedding it through `llm-gateway-sdk`, discovering categories, and training classifiers against those discovered categories. Domain profiles compose with language policies; Arabic and English can share categories in one aligned multilingual embedding space.

## Install and configure

```sh
uv pip install --python .venv/bin/python -e '.[dev]'
# Optional UMAP reducer:
uv pip install --python .venv/bin/python -e '.[discovery]'
```

Configure the gateway using `.env.example`. Set the correct embedding dimensions for the deployment, or leave dimensions unset to learn and persist the returned width. Set `EMBEDDING_REVISION` to an immutable deployment/model revision. Provider, model, endpoint, dimensions, task and profile fingerprint identify the fitted representation; changing its identity requires refitting. Mutable gateway aliases must have their revision changed when repointed.

The default `configs/pipeline_config.yaml` uses HDBSCAN, PCA, and a shared English/Arabic policy. See [profiles.example.yaml](configs/profiles.example.yaml) for claims, technical incidents, and customer service with custom columns, routes, metadata and additional languages. Unsupported configuration keys or strategy names fail validation.

## Fit your data

```sh
ticket-clf --input claims.jsonl \
  --config configs/profiles.example.yaml --domain-id customer_claim \
  --no-llm-naming --report-path claims-report.md
```

Example input records for that profile:

```json
{"claim_id":"c1","narrative":"My payment failed and I need a refund","conversation_id":"thread-1","language":"en","amount":100,"product":"card"}
{"claim_id":"c2","narrative":"حصلت مشكلة في دفع الفلوس ومحتاج استرجع المبلغ","conversation_id":"thread-2","language":"ar","amount":100,"product":"card"}
```

Use a representative dataset for training; these two records illustrate the schema. Labels are optional. Conversation IDs should link translations/paraphrases of the same example so they remain in one evaluation group. Exact normalized duplicates are grouped automatically. Explicit `train`, `val`, `test` splits are accepted and checked for duplicate/conversation overlap. Temporal splitting additionally validates chronology. Very small datasets can fit without producing a deployable classifier; the result states that evaluation was unavailable and queues inference for review.

For a benchmark, use `--dataset textoir_banking77` (also CLINC, IT tickets, CFPB claims and incident samples). Omit `--top-n-classes` and K bounds for the full category set and dataset-driven discovery. Set `--max-samples` when intentionally limiting a run. CLI values override YAML for strategy and K bounds. Explicit K bounds select the K-Means baseline unless `--strategy` is supplied.

```sh
ticket-clf --dataset textoir_banking77 --strategy aligned_kmeans --no-llm-naming
ticket-clf --dataset textoir_clinc_oos --strategy hdbscan_density \
  --oos-file domain-unknown-examples.json --no-llm-naming
```

These commands call the configured embedding deployment. Clustering comparisons reuse content-addressed embeddings. The OOS file is an optional JSON array of domain-appropriate probe strings; ordinary household or weather inquiries are not universally out of scope.

## API

Run `uvicorn ticket_classification.controllers.api:create_app --factory`.

- `POST /api/v1/fit`: `records`, optional `domain_id`, `strategy`, `provider_override`, `use_llm_naming` and `oos_texts`.
- `POST /api/v1/classify`: `ticket_id`, `text`, optional `metadata`, `domain_id`, `taxonomy_version`, `language_hint`, `display_language`, `provider_override` and `enable_llm_reranker`.
- `POST /api/v1/classify/batch`: a `tickets` array with those classification fields.
- `GET /api/v1/clusters?domain_id=customer_claim`: categories, stable IDs, active taxonomy version and lineage.
- `GET /api/v1/review-items?domain_id=customer_claim`: persisted masked items awaiting review or a later fit.

Prediction status is `known`, `unknown`, or `needs_review`. Unknown intent returns cluster `-1` and no stable category/route ID. Empty inputs and unsupported languages are review cases, separate from semantic novelty. A language hint is retained as source metadata; it does not override uncertain detection. Short text may have undetermined language and require review.

All generated routes must belong to the profile's approved route map. Category identity is independent of display language; localized names are used where supplied by naming output, otherwise the primary name is returned. Changing display language does not retrain or renumber categories. The LLM reranker can reject a category or resolve an ambiguous supported class; it cannot override the geometric unknown gate. Turning naming off does not disable embeddings.

## Discovery and model lifecycle

HDBSCAN estimates category count without an explicit K. Candidate scoring combines original embedding-space silhouette, accepted coverage, support and resampling stability. It can return one coherent group or all noise. The K-Means alternative searches a dataset-size/support-derived feasible interval, refines candidates within a compute budget, and records boundaries and budget exhaustion. K is a model selection result, not a claimed recovery of a unique business taxonomy.

PCA, optional UMAP and no reduction are supported. Fitted reducers are retained; inference never refits them. Discovery assignments preserve noise. Frozen centroid prediction is an explicitly separate baseline, not HDBSCAN `approximate_predict`; new categories require a new discovery run. Keyword extraction uses class-based TF-IDF. The legacy `medoid_texts` field contains centroid-nearest representatives.

The classifier suite includes majority, logistic regression, calibrated SVM, gradient boosting and a direct prototype comparison. Rare classes are excluded without silently relaxing support thresholds; supported single-class data uses a constant classifier. The novelty threshold uses held-out supported examples when available and explicitly reports a training fallback otherwise. Evaluation and serving use the same distance-space gate. Cluster agreement, mapped ground-truth metrics and per-language coverage are reported separately.

Each successful fit creates an immutable `versions/<taxonomy_version>.joblib` snapshot and atomically updates the active bundle. Generic-domain artifacts remain under `data/artifacts/`; other domains use `data/artifacts/<domain_id>/`. Reload restores the fitted configuration, language policy, features, taxonomy and embedding identity. Failed refits restore the previous in-memory model. Requests with a stale active taxonomy version are rejected. Pre-review unversioned artifacts require refitting.

Stable IDs are matched across compatible embedding spaces, with separate birth/retirement/split/merge diagnostics when K changes. Similarity-based lineage is a heuristic and should be reviewed before relying on it for business category continuity.

## Language extensions and validation

The Unicode default preserves non-Latin text, negation and technical codes. Arabic normalization, localized boilerplate, sensitive-data regexes, stopwords, preserved tokens, chunk size and metadata schema are configurable. CJK text uses a character fallback until a specialized tokenizer is selected. Long narratives use cached chunk embeddings and a length-weighted aggregate; a single primary category is returned.

Applications can register detectors, tokenizers and lexical postprocessors through `domain.cleaning.languages`, and discovery engine factories through `domain.clustering.strategies`. Register plugins at startup before fitting or reloading; artifacts retain plugin names. Specialized discovery factories must implement the `ClusteringEngine` contract, including diagnostics and persistence-compatible state.

Run `make ci-quality` and `make ci-docs`. Tests include Arabic/English/mixed-script preservation, cross-language category agreement using synthetic semantic vectors, grouping, domain isolation, model reloads, unknown queues, failed-refit recovery, embedding contract validation and discovery edge cases. These tests establish software behavior, **not the multilingual accuracy of a real embedding deployment**. Full multi-domain quality, dialect coverage, translation quality and human-reviewed routing usefulness require representative data and deployed-model evaluation. Translation and per-language taxonomies are not enabled strategies in this implementation; use the shared representation or separate configured domain profiles.

The [repository comparison](REPOSITORY_COMPARISON.md) records the research and the original gaps. The earlier [benchmark report](BENCHMARK_REPORT.md) predates these changes and should not be used as current validation evidence.
