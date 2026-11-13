# Production deployment

## Supported topology

Run the HTTP API and training consumer as separate processes on one Linux host. They share a local persistent data directory and identical configuration and credentials. SQLite WAL and process file locks coordinate jobs and releases. Do not place these databases on NFS or use this storage implementation across hosts. A multi-host deployment requires a shared transactional job/review store and an object-store-backed model registry.

The supplied container runs as UID/GID 10001, with a read-only root filesystem, a writable data volume and temporary storage. The Compose deployment binds the API to localhost for a TLS reverse proxy and applies independent CPU, memory and process limits to API and worker services. Adjust those limits using measurements of your dataset, vocabulary and embedding width.

## Initial setup

Install Docker Engine with Compose. From the application directory:

```sh
cp -n .env.example .env
uv run --frozen python scripts/create_secrets.py --subject operations-admin
```

The setup command prompts for an issued gateway API key without echoing it. Alternatively, supply `--gateway-key-file /secure/path/gateway-key`. It creates a new `secrets` directory and refuses to overwrite an existing one. The admin token is stored as the key of the mapping in `secrets/auth_tokens.json`; distribute it through your secret manager, not logs or source control. Add separate least-privilege reader and trainer principals as needed.

The directory is accessible only to its owner. Files are readable for container secret mounts; the directory protects them from other host users. For managed deployments, mount equivalent files from the platform's secret store rather than generating credentials on the application host.

Edit `.env`; merge the example settings if that file already exists:

| Setting | Required value |
| --- | --- |
| `LLM_GATEWAY_BASE_URL` | Your HTTPS gateway endpoint. |
| `LLM_GATEWAY_DEFAULT_EMBEDDING_MODEL` | An aligned multilingual deployment or gateway alias. |
| `LLM_GATEWAY_EMBEDDING_DIMENSIONS` | Its returned width; do not assume 1536 for every deployment. |
| `EMBEDDING_REVISION` | An immutable deployment/model revision. |
| `DOMAIN_ID` / `TENANT_ID` | The profile and tenant whose model are required for readiness. |
| `ALLOWED_HOSTS` | A JSON list of accepted host names, including the reverse proxy's public host and health-probe host. |

Keep `APP_ENV=production`, `AUTH_ENABLED=true`, certificate verification enabled and SDK retries disabled. Production startup rejects absent/short authentication and artifact-signing secrets, missing subject names, invalid principal scopes, an absent embedding revision, a missing configuration file, relative data paths and an insecure gateway connection. Interactive API docs are disabled by default.

Compose mounts the credential files and fixes configuration/data paths inside the container. Configure your domains and release policy in `configs/pipeline_config.yaml` before building, or mount your own pipeline file at `/app/configs/pipeline_config.yaml` read-only in both services. Rebuild or recreate both services when changing configuration or credentials.

```sh
docker compose build
docker compose up -d
docker compose ps
curl --fail http://127.0.0.1:8000/health
```

Keep the API behind a TLS reverse proxy with request/body/time limits. The bundled server ignores forwarded headers, does not emit a server header, limits concurrent connections and gives shutdown a bounded grace period. The reverse proxy should enforce HTTPS and HSTS. The API returns a request ID, `nosniff` and no-store response headers.

Before the first model is activated, `/health` returns 200 and `/health/ready` returns 503. Submit fitting and approval requests through the local management path while bootstrapping. Route classification traffic to the instance only after readiness succeeds. See the [user guide](user-guide.md) for requests and approval.

## Native processes

A native POSIX installation can use the same secret files and configuration:

```sh
uv sync --frozen --no-dev
export APP_ENV=production AUTH_ENABLED=true
export CONFIG_PATH="$PWD/configs/pipeline_config.yaml" DATA_DIR="$PWD/data"
export AUTH_TOKENS_FILE="$PWD/secrets/auth_tokens.json"
export ARTIFACT_HMAC_SECRET_FILE="$PWD/secrets/artifact_hmac"
export GATEWAY_KEY_FILE="$PWD/secrets/gateway_key"
uv run --frozen ticket-api --workers 1 --limit-concurrency 64
```

Run `uv run --frozen ticket-worker` in a separate supervised process with the same environment. Both commands also read `.env`. Use your process supervisor to enforce CPU/memory limits and restart failures. Optional UMAP support requires installation with `--extra discovery` natively, or `docker compose build --build-arg INSTALL_DISCOVERY=true` for the container. Custom plugins and numerical dependencies must match between fitting and serving.

## Jobs and recovery

Authenticated `/api/v1/fit` always enqueues a durable job and returns HTTP 202. The API never trains inline in production. Requests have configurable body, text, metadata, batch and dataset limits. Body limits are enforced as data arrives, including chunked requests. Configure upstream upload and idle timeouts as well.

Workers claim jobs transactionally and renew a lease while fitting. Each attempt gets a unique ownership token; cancelled or stale attempts cannot commit checkpoints, overwrite results or publish a model. Queued jobs survive API restarts. A killed worker's running job becomes claimable after its lease expires (60 seconds by default). After three interrupted attempts, the job fails instead of repeatedly consuming resources; `FIT_JOB_MAX_ATTEMPTS` controls that limit. Explicitly resubmitting a failed/cancelled job resets it for another attempt.

Validated embedding batches are cached immediately. A completed candidate checkpoint is bound to the job, input, configuration, representation, numerical environment and checkpoint revision. Recovery reuses that candidate without repeating training, including a crash after release publication but before the completion update. Before the candidate checkpoint, clustering/classification rerun while completed embedding batches are reused. Execution is at least once, with fenced publication.

The worker handles SIGTERM by stopping new claims after its current job. If shutdown exceeds the supervisor/container grace period, process termination leaves the job available for lease-based recovery. Numerical operations may finish before observing cancellation; size worker resources independently from serving resources.

## Release policy

Fitting saves an immutable candidate. A completed job can have `publication.status=candidate`; inspect `release_gate.blockers` before attempting activation. Failed or pending candidates preserve the active model.

Defaults require held-out evaluation, sufficient held-out calibration, evidence for every supported language, per-language cluster agreement macro F1 of at least 0.6, and accepted coverage of at least 0.5. Taxonomy births, splits, merges and route changes require approval by an authenticated admin. Human approval cannot override failed quality checks.

Configure sample requirements and business thresholds for the deployment:

```yaml
release:
  require_evaluation: true
  require_calibration: true
  min_language_samples: 30
  min_language_macro_f1: 0.8
  min_language_accepted_coverage: 0.8
  min_ground_truth_macro_f1: 0.85
  min_oos_recall: 0.8
  max_known_false_positive_rate: 0.05
  require_taxonomy_approval: true
```

These are example thresholds; validate them for your routing task. The last three numerical thresholds are optional and fail closed on missing evidence when configured. Business macro F1 uses held-out mapped ground truth for the selected model, independently of agreement with discovered pseudo-labels. Include minority languages, dialects, code switching and unknown intents in the release corpus. Keep translations, near duplicates and conversations together when splitting data.

## Health, logs and monitoring

| Endpoint/probe | Purpose |
| --- | --- |
| `/health` | API process liveness; used by the container health check. |
| `/health/ready` | Required model is loaded and its representation matches the deployment. Production cannot bypass this check with `require_model=false`. |
| `/metrics` | Tenant-specific in-process metrics for an admin with access to all domains in that tenant. |
| Worker container probe | Recent consumer/lease heartbeat; no model or gateway initialization. |

API and worker write JSON logs to stdout. Request events contain a generated request ID, method, route template, status and duration. Bodies, authorization headers, query strings and ticket identifiers are not included in request logs. Avoid placing sensitive values in job identifiers or custom plugin log messages.

Monitor HTTP error/latency rates, readiness, gateway throttling/retries, per-language review/unknown rates, active model versions, review backlog, job age/status, process memory, volume capacity and worker health. API metrics are per process and reset on restart; collect them from every replica with your monitoring system. Worker telemetry is separate from HTTP-process counters. Container health status alone does not automatically restart an unhealthy process; configure supervisor/monitoring alerts accordingly.

Gateway request budgets are per adapter/process. Apply global quotas and spend limits on the issued gateway key. Scaling processes or domains does not create a distributed application-wide quota.

## Backups, upgrades and rollback

Persist the `ticket-data` volume. It contains immutable artifacts, release records, embedding cache, review data, jobs and checkpoints. Back up the artifact signing key separately in your secret store; changing it invalidates existing artifact signatures. Never accept uploaded joblib artifacts from untrusted sources.

For a consistent backup:

1. Stop new submissions and stop API/worker services. Allow jobs to finish, or rely on lease recovery after a forced stop.
2. Back up the entire stopped volume, configuration, image identifier and secret-store references. Preserve SQLite database/WAL files together.
3. Restore into an isolated deployment and verify job/review counts, active model loading and a representative classification request.

For online SQLite backups, use SQLite's backup API for each database and coordinate the release pointer/artifact snapshot. Copying only a live `.sqlite` file is insufficient.

Container CI builds the runtime image, verifies non-root execution and startup validation, and audits catalogued runtime dependencies. The Git-pinned gateway SDK is outside PyPI advisory coverage and requires separate source review when updating its revision.

Build and retain immutable image tags for upgrades. Run validation in a separate data volume, then stop writers before taking a backup and changing the deployed image. Retain the previous image and full volume backup for application rollback; avoid running an older binary against a newly migrated database without validation.

For a model rollback, POST `/api/v1/models/rollback` with a permitted domain and optional historical version. Previously approved releases retain their approval. Verify the active version, readiness and representative predictions afterward. Changing the embedding deployment also requires restoring its matching revision or fitting a compatible model.

The active model pointer is `release_record.json`. Candidate files and manifests live in `versions/`; non-default tenants use `artifacts/tenants/<tenant>/`, with a domain directory outside `generic`. Consumers must use the registry API rather than read the legacy `model_registry.joblib` path. Artifacts without tenant provenance are never deserialized or assigned to a tenant automatically. They remain inactive while the control plane accepts a replacement fit; a validated replacement can be persisted and reloaded using the normal scoped path. Genuinely foreign-tenant artifacts still fail isolation checks. Artifacts without tenant provenance or release evidence require rebuilding before promotion.
