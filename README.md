# Ticket and claim classification

Discover categories in historical tickets and classify new text against the resulting taxonomy. The pipeline cleans and masks text, builds multilingual embeddings, discovers clusters with a dataset-dependent category count, and trains a classifier. Domain profiles configure input fields, supported languages, metadata and routing queues for customer service, technical incidents, claims or other industries.

## Installation

Python 3.11 or later and [uv](https://docs.astral.sh/uv/) are required for a native installation:

```sh
uv sync --all-groups --frozen
```

For container deployments, follow the [production setup](docs/production-runtime.md). It runs separate API and fit-worker processes with persistent storage and mounted secrets.

## Configure your domain

Start with [pipeline_config.yaml](configs/pipeline_config.yaml), or adapt [profiles.example.yaml](configs/profiles.example.yaml). Set the input text and identifier columns, supported languages and approved routing queues. The configured embedding deployment must align all supported languages in one vector space. Set `EMBEDDING_REVISION` to an immutable deployment revision and refit whenever that deployment changes.

Example records for the claims profile:

```json
{"claim_id":"c1","narrative":"My payment failed and I need a refund","conversation_id":"thread-1","language":"en","amount":100,"product":"card"}
{"claim_id":"c2","narrative":"حصلت مشكلة في دفع الفلوس ومحتاج استرجع المبلغ","conversation_id":"thread-2","language":"ar","amount":100,"product":"card"}
```

These records illustrate the schema. Training and release evaluation require a representative dataset. Labels are optional; conversation IDs should group translations and paraphrases so they remain in the same evaluation partition.

## Fit and classify

For a native batch fit with your configured environment:

```sh
uv run --frozen ticket-clf --input claims.jsonl \
  --config configs/profiles.example.yaml --domain-id customer_claim \
  --no-llm-naming --report-path data/reports/claims.md
```

Fitting calls the configured embedding gateway even when naming is disabled. HDBSCAN discovers category count directly; the K-Means strategy selects K from feasible candidates within a work budget. Fitting saves a candidate version. Quality checks and any required taxonomy approval must pass before that version becomes active.

The API provides fit submission, job status and cancellation, single and batch classification, taxonomy inspection and a review queue. See the [user guide](docs/user-guide.md) for authentication, request examples, language behavior and release approval.

Predictions have one of three statuses:

| Status | Meaning |
| --- | --- |
| `known` | A supported match with an approved routing queue. |
| `unknown` | An unfamiliar intent without an assigned category or route. |
| `needs_review` | Empty text, unsupported or uncertain language, insufficient class evidence, or an ambiguous match. |

Category IDs remain independent of display language. New categories require another fit; review items are not incorporated automatically. The app returns routing information; delivery to another ticketing system belongs to your integration.

## Operations

Read the [production guide](docs/production-runtime.md) for deployment, secrets, readiness, monitoring, backup and rollback. The bundled storage supports multiple processes on a single POSIX host. Multiple hosts require shared transactional storage and an object-store-backed registry.

Automated tests use synthetic embeddings to verify application behavior. Evaluate real routing accuracy, dialects, minority languages and unknown-intent handling on representative held-out data before activating a production model.
