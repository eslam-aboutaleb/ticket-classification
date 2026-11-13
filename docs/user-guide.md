# Using the classification API

## Authentication

Send an issued token using `Authorization: Bearer <token>` or `X-API-Key: <token>`. Tokens identify a subject, tenant, roles and allowed domains. Readers classify tickets and inspect permitted taxonomies and review items. Trainers also submit and cancel fits and adjudicate review items. Admins approve taxonomy changes and access tenant operational metrics. A caller cannot select another tenant through a request parameter.

The examples below use `TICKET_API_URL` and `TICKET_API_TOKEN` from your client environment:

```sh
export TICKET_API_URL=http://127.0.0.1:8000
```

Use HTTPS for connections outside the deployment host. Interactive API documentation is disabled by default; an operator can enable it with `API_DOCS_ENABLED=true`.

## Submit historical data

The default profile expects `ticket_id` and `raw_text`. A custom profile can map different columns and add metadata. A `records.json` file contains a JSON array of historical records; labels are optional.

```sh
python - <<'PY'
import json
from pathlib import Path
records = json.loads(Path('records.json').read_text())
Path('fit-request.json').write_text(json.dumps({
    'records': records, 'domain_id': 'generic',
    'use_llm_naming': False, 'idempotency_key': 'historical-import-001',
}))
PY
curl --fail-with-body "$TICKET_API_URL/api/v1/fit" \
  -H "Authorization: Bearer $TICKET_API_TOKEN" \
  -H 'Content-Type: application/json' --data-binary @fit-request.json
```

Authenticated fit requests return HTTP 202 with `job_id`. Reusing an idempotency key returns the existing active or completed job. Use a new key when intentionally fitting different data. Fit strategy, naming, provider override and optional unknown-intent probe texts (`oos_texts`) are accepted in the same request. `async_job=false` does not bypass the durable queue in authenticated deployments.

```sh
curl --fail-with-body "$TICKET_API_URL/api/v1/jobs/<job-id>" \
  -H "Authorization: Bearer $TICKET_API_TOKEN"
curl --fail-with-body -X POST "$TICKET_API_URL/api/v1/jobs/<job-id>/cancel" \
  -H "Authorization: Bearer $TICKET_API_TOKEN"
```

Job results include a candidate taxonomy version, release evidence and `publication.status`. `completed` means fitting completed; the result can still be a candidate awaiting sufficient quality evidence or approval. Cancellation prevents a cancelled attempt from publishing, but a native numerical operation may continue until the next cancellation check.

## Activate a candidate

Inspect the fit result's `release_gate.blockers`. A candidate with failed quality checks cannot be forced into service by approval. Correct the dataset or policy and fit a new version. Each candidate retains its evaluated release policy and evidence.

An admin approves an eligible taxonomy through:

```sh
curl --fail-with-body "$TICKET_API_URL/api/v1/models/promote" \
  -H "Authorization: Bearer $TICKET_API_TOKEN" -H 'Content-Type: application/json' \
  --data '{"domain_id":"generic","target_version":"<candidate-version>","expected_previous_version":"<active-version>","approve_taxonomy":true}'
```

Use an empty `expected_previous_version` for the first activation. A stale expected version returns HTTP 409. Approval records use the authenticated subject. By default, births, splits, merges and route changes require approval. Otherwise eligible candidates without such changes can activate automatically.

## Classify tickets

```sh
curl --fail-with-body "$TICKET_API_URL/api/v1/classify" \
  -H "Authorization: Bearer $TICKET_API_TOKEN" -H 'Content-Type: application/json' \
  --data '{"ticket_id":"live-001","text":"My payment failed and I need help","domain_id":"generic"}'
```

Optional fields include metadata, language hint, display language, provider override, reranker selection and an active `taxonomy_version` assertion. A stale asserted version is rejected. Responses include status, category label, stable category ID, routing queue, language evidence, confidence, distance and taxonomy version.

Batch requests preserve input order:

```sh
curl --fail-with-body "$TICKET_API_URL/api/v1/classify/batch" \
  -H "Authorization: Bearer $TICKET_API_TOKEN" -H 'Content-Type: application/json' \
  --data '{"tickets":[{"ticket_id":"one","text":"My payment failed"},{"ticket_id":"two","text":"عندي مشكلة في دفع الفلوس"}]}'
```

Compatible items share embedding and classifier batches. All generated routes must belong to the domain's approved route map. Optional LLM reranking cannot override the geometric unknown-intent gate.

## Languages and domain profiles

Supported languages use one shared taxonomy and an aligned multilingual embedding model. Keep supported-language settings consistent with your evaluated model coverage. Language hints are retained as source metadata and do not override uncertain detection. Short messages and code switching need representative evaluation. The Unicode tokenizer preserves non-Latin scripts and technical codes; CJK text uses a character fallback unless a specialized tokenizer is configured.

Profiles support custom text/ID/group/timestamp columns, temporal or grouped splits, PII patterns, boilerplate, stopwords, Arabic normalization, chunk size, metadata schemas and approved routes. Long narratives use cached chunk embeddings and a length-weighted aggregate. The current API returns one primary category per item.

Applications can register additional language detectors, tokenizers, lexical transforms and discovery strategies before fitting or loading models. Deploy the same plugin registration and dependencies in API and worker processes. Changing a model, semantic normalization or embedding revision requires refitting. A display-language change does not renumber categories.

## Inspect categories and review uncertain tickets

```sh
curl --fail-with-body "$TICKET_API_URL/api/v1/clusters?domain_id=generic" \
  -H "Authorization: Bearer $TICKET_API_TOKEN"
curl --fail-with-body "$TICKET_API_URL/api/v1/review-items?domain_id=generic&limit=50" \
  -H "Authorization: Bearer $TICKET_API_TOKEN"
```

Review items contain masked text, reasons and model references. Use the returned cursor for subsequent pages. Repeated occurrences update the existing business ticket rather than create a separate review item for each model version.

```sh
curl --fail-with-body "$TICKET_API_URL/api/v1/review-items/live-001/adjudicate" \
  -H "Authorization: Bearer $TICKET_API_TOKEN" -H 'Content-Type: application/json' \
  --data '{"domain_id":"generic","status":"approved","adjudicated_label":"payment_issue","approved_for_training":true,"notes":"Reviewed against the customer record"}'
```

Review approval and consent are recorded; they do not trigger an automatic retraining run. Export and curate approved feedback into a later fit according to your data-retention policy.

## Errors

| HTTP status | Action |
| --- | --- |
| 400 / 422 | Correct the request, profile or representation mismatch. |
| 401 / 403 | Check credentials, role and domain scope. |
| 404 | Check the job or review identifier and tenant. |
| 409 | Refresh the active version or resolve release blockers. |
| 413 | Reduce payload size. |
| 429 | Wait for workload capacity before submitting new work. |
| 503 | The service is starting, lacks a valid required model, or is at connection capacity. |

Every HTTP response carries `X-Request-ID` for operational correlation. Provide that identifier when reporting a failed request; do not include authentication tokens or unmasked ticket contents.

## Run a full benchmark

`make ci-benchmark` uses all 13,083 Banking77 input records and all 77 labels. It prepares 900 external CLINC150 test probes, passes them with `--oos-file`, and writes reports under `data/benchmarks/banking77-full/`. The source revision, file hashes, cohort labels and probe license are retained alongside the results. This command calls the configured embedding gateway.

The benchmark keeps the official test partition. Eight normalized duplicate copies in earlier partitions are removed by the explicit `cross_split_duplicates: prefer_holdout` policy; ordinary application profiles retain the strict `error` policy. This is leakage cleanup, not a dataset/class cap. The discovery sample ceiling exceeds the full training partition, so all eligible training rows participate in discovery after the normal empty-text and language checks. Category count remains dataset-driven.

The external challenge set contains credit/insurance/retirement intents outside the 77 Banking77 labels and clearly nonfinancial intents. It is curated from [CLINC150](https://github.com/clinc/oos-eval), not an official Banking77 OOS split or proof of multilingual performance. Probe membership is fixed before scoring. No OOS test scores select the threshold.

The report compares the existing 96th-percentile distance threshold with a 99th-percentile threshold fitted only on known validation examples. It reports known false positives over the entire known test cohort, OOS recall and AUROC. A higher threshold can reduce false positives while sacrificing OOS recall; examine both quantities. The benchmark config trains majority, logistic regression and prototype candidates, disables optional LLM naming/reranking, and records these choices.

To prepare and run explicitly:

```sh
uv run --frozen python scripts/prepare_banking77_benchmark.py
DATA_DIR="$PWD/data/benchmarks/banking77-full/run-data" uv run --frozen ticket-clf \
  --dataset textoir_banking77 --config data/benchmarks/banking77-full/benchmark.yaml \
  --domain-id banking77 --provider litellm --no-llm-naming \
  --oos-file data/benchmarks/banking77-full/oos-test.json \
  --report-path data/benchmarks/banking77-full/full-benchmark.md \
  > data/benchmarks/banking77-full/full-benchmark.json
```

For another domain, provide its own OOS file and configuration using `OOS_FILE`, `BENCHMARK_CONFIG`, `BENCHMARK_DOMAIN` and `BENCHMARK_DATASET` Make variables. Do not reuse Banking77's OOS labels without checking that they are truly unknown for the new domain.
