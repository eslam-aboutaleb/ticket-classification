"""HTTP API routes, service wiring and application lifecycle."""

from __future__ import annotations

import logging
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Annotated, Any

import pandas as pd
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ticket_classification.config.pipeline import PipelineConfig
from ticket_classification.config.settings import AppSettings, LLMProvider
from ticket_classification.controllers.auth import (
    Principal,
    require_domain_access,
    require_roles,
    resolve_principal,
)
from ticket_classification.controllers.operations import OperationsMiddleware
from ticket_classification.controllers.request_limits import RequestSizeLimitMiddleware
from ticket_classification.gateways.llm_gateway_adapter import (
    BaseCloudGateway,
    LLMGatewaySDKAdapter,
)
from ticket_classification.models.schemas import (
    BatchClassifyRequest,
    ClassifyTicketRequest,
    ClassifyTicketResponse,
    FitJobResponse,
    FitRequest,
    PromoteModelRequest,
    ProviderStatusResponse,
    ProviderSwitchRequest,
    ReviewAdjudicationRequest,
    RollbackModelRequest,
)
from ticket_classification.repositories.dataset_repo import TicketDatasetRepository
from ticket_classification.repositories.embedding_cache_repo import (
    EmbeddingCacheRepository,
)
from ticket_classification.repositories.fit_job_repo import FitJobRepository
from ticket_classification.repositories.model_registry_repo import (
    ModelRegistryRepository,
)
from ticket_classification.services.pipeline_service import TicketPipelineService
from ticket_classification.services.telemetry import TelemetryCollector
from ticket_classification.views.report_view import ReportAndPlotView


class ServiceContainer:
    """Singleton Dependency Injection container wiring Gateways, Repositories, Services, and Views."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        gateway: BaseCloudGateway | None = None,
        max_cached_services: int = 32,
    ) -> None:
        self.settings = settings or AppSettings()
        self.telemetry = TelemetryCollector()
        self._tenant_telemetry = {self.settings.tenant_id: self.telemetry}
        self.pipeline_config = PipelineConfig.model_validate(
            self.settings.load_pipeline_yaml()
        )
        self.gateway = gateway or LLMGatewaySDKAdapter(
            self.settings,
            max_concurrency=self.pipeline_config.embedding.max_concurrency,
            max_requests_per_minute=self.pipeline_config.admission.provider_request_budget_per_minute,
            telemetry=self.telemetry,
        )
        self._injected_gateway = gateway
        self._services: OrderedDict[tuple[str, str], TicketPipelineService] = (
            OrderedDict()
        )
        self._max_cached_services = max(1, max_cached_services)
        self._lock = RLock()
        self.fit_job_repo = FitJobRepository(
            Path(self.settings.data_dir) / "fit_jobs.sqlite"
        )
        self.pipeline_service = self.get_service(
            self.settings.domain_id, tenant_id=self.settings.tenant_id
        )
        self.dataset_repo = self.pipeline_service.dataset_repo
        self.cache_repo = self.pipeline_service.cache_repo
        self.model_registry_repo = self.pipeline_service.model_registry_repo
        self.report_view = ReportAndPlotView(
            str(Path(self.settings.data_dir) / "artifacts")
        )

    def get_service(
        self,
        domain_id: str | None = None,
        tenant_id: str | None = None,
    ) -> TicketPipelineService:
        domain_id = domain_id or self.settings.domain_id
        tenant_id = tenant_id or self.settings.tenant_id
        key = (tenant_id, domain_id)
        with self._lock:
            if key in self._services:
                self._services.move_to_end(key)
                return self._services[key]
            config = PipelineConfig.model_validate(self.settings.load_pipeline_yaml())
            config.profile(domain_id)  # Validate before constructing any storage paths.
            domain_settings = (
                self.settings
                if domain_id == self.settings.domain_id
                and tenant_id == self.settings.tenant_id
                else self.settings.model_copy(
                    update={"domain_id": domain_id, "tenant_id": tenant_id}, deep=True
                )
            )
            cache_path = Path(config.embedding.cache_db_path)
            if not cache_path.is_absolute():
                cache_path = Path(self.settings.data_dir) / cache_path
            telemetry = self._tenant_telemetry.setdefault(
                tenant_id, TelemetryCollector()
            )
            gateway = self._injected_gateway or LLMGatewaySDKAdapter(
                domain_settings,
                max_concurrency=config.embedding.max_concurrency,
                max_requests_per_minute=config.admission.provider_request_budget_per_minute,
                telemetry=telemetry,
            )
            service = TicketPipelineService(
                settings=domain_settings,
                gateway=gateway,
                dataset_repo=TicketDatasetRepository(
                    self.settings.data_dir, config.datasets
                ),
                cache_repo=EmbeddingCacheRepository(cache_path),
                model_registry_repo=ModelRegistryRepository.for_scope(
                    self.settings.data_dir,
                    hmac_secret=domain_settings.artifact_hmac_secret,
                    tenant_id=tenant_id,
                    domain_id=domain_id,
                    enforce_release_gate=True,
                ),
                telemetry=telemetry,
                fit_job_repo=self.fit_job_repo,
                tenant_id=tenant_id,
            )
            self._services[key] = service
            while len(self._services) > self._max_cached_services:
                _, evicted = self._services.popitem(last=False)
                if evicted.gateway is not self._injected_gateway:
                    evicted.gateway.close()
            return service

    def submit_fit_job(
        self,
        tenant_id: str,
        domain_id: str,
        payload: FitRequest,
    ) -> dict[str, Any]:
        """Persist work for the independent lease-based fit worker."""
        self.get_service(domain_id, tenant_id=tenant_id)
        job_payload = {
            "records": payload.records,
            "strategy": payload.strategy,
            "use_llm_naming": payload.use_llm_naming,
            "provider_override": payload.provider_override,
            "oos_texts": payload.oos_texts,
        }
        job, _ = self.fit_job_repo.enqueue(
            tenant_id=tenant_id,
            domain_id=domain_id,
            payload=job_payload,
            idempotency_key=payload.idempotency_key,
            max_concurrent_jobs=self.pipeline_config.admission.max_concurrent_jobs_per_tenant,
        )

        return job

    def close(self) -> None:
        with self._lock:
            for svc in self._services.values():
                if svc.gateway is not self._injected_gateway:
                    svc.gateway.close()
        if self.gateway is not self._injected_gateway:
            self.gateway.close()


@lru_cache(maxsize=1)
def get_container() -> ServiceContainer:
    return ServiceContainer()


ContainerDep = Annotated[ServiceContainer, Depends(get_container)]


def get_principal(
    container: ContainerDep,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> Principal:
    return resolve_principal(
        settings=container.settings,
        authorization=authorization,
        x_api_key=x_api_key,
    )


PrincipalDep = Annotated[Principal, Depends(get_principal)]

router = APIRouter(prefix="/api/v1")


@router.get("/provider", response_model=ProviderStatusResponse)
def get_provider_status(
    container: ContainerDep,
    principal: PrincipalDep,
) -> ProviderStatusResponse:
    """Inspect active cloud provider ('litellm', 'openai', or 'gemini') and model tier mappings."""
    require_roles(principal, "reader", "trainer", "admin")
    active = container.gateway.get_active_provider()
    info = container.pipeline_service.switch_provider(active)
    return ProviderStatusResponse(**info)


@router.post("/provider", response_model=ProviderStatusResponse)
def switch_provider_endpoint(
    payload: ProviderSwitchRequest,
    container: ContainerDep,
    principal: PrincipalDep,
) -> ProviderStatusResponse:
    """Dynamically switch active provider between 'litellm', 'openai', and 'gemini'."""
    require_roles(principal, "trainer", "admin")
    try:
        info = container.pipeline_service.switch_provider(payload.provider)
        return ProviderStatusResponse(**info)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported provider '{payload.provider}'. Choose from: {[p.value for p in LLMProvider]}",
        ) from exc


@router.get("/clusters")
def list_discovered_clusters(
    container: ContainerDep,
    principal: PrincipalDep,
    domain_id: str | None = None,
) -> dict[str, Any]:
    """Return discovered cluster profiles and taxonomy metadata."""
    require_roles(principal, "reader", "trainer", "admin")
    target_domain = domain_id or container.settings.domain_id
    require_domain_access(principal, target_domain)
    try:
        service = container.get_service(target_domain, tenant_id=principal.tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return service.inspect_taxonomy()


@router.post("/classify", response_model=ClassifyTicketResponse)
def classify_ticket_endpoint(
    payload: ClassifyTicketRequest,
    container: ContainerDep,
    principal: PrincipalDep,
) -> ClassifyTicketResponse:
    """Classify a single support ticket or claim against discovered clusters."""
    require_roles(principal, "reader", "trainer", "admin")
    target_domain = payload.domain_id or container.settings.domain_id
    require_domain_access(principal, target_domain)
    try:
        res = container.get_service(
            target_domain, tenant_id=principal.tenant_id
        ).classify_single_ticket(
            ticket_id=payload.ticket_id,
            raw_text=payload.text,
            metadata=payload.metadata,
            provider_override=payload.provider_override,
            language_hint=payload.language_hint,
            display_language=payload.display_language,
            taxonomy_version=payload.taxonomy_version,
            enable_llm_reranker=payload.enable_llm_reranker,
        )
        return ClassifyTicketResponse(**asdict(res))
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/classify/batch", response_model=list[ClassifyTicketResponse])
def classify_batch_endpoint(
    payload: BatchClassifyRequest,
    container: ContainerDep,
    principal: PrincipalDep,
) -> list[ClassifyTicketResponse]:
    """Classify a batch of tickets using true compatible grouping and batched matrix inference."""
    require_roles(principal, "reader", "trainer", "admin")
    grouped_indices: dict[tuple[str, str | None], list[int]] = {}
    for idx, item in enumerate(payload.tickets):
        target_domain = item.domain_id or container.settings.domain_id
        require_domain_access(principal, target_domain)
        prov = payload.provider_override or item.provider_override
        grouped_indices.setdefault((target_domain, prov), []).append(idx)

    ordered: list[ClassifyTicketResponse | None] = [None] * len(payload.tickets)
    try:
        for (target_domain, prov), indices in grouped_indices.items():
            service = container.get_service(
                target_domain, tenant_id=principal.tenant_id
            )
            batch_items = [
                {
                    "ticket_id": payload.tickets[i].ticket_id,
                    "raw_text": payload.tickets[i].text,
                    "metadata": payload.tickets[i].metadata,
                    "provider_override": prov,
                    "enable_llm_reranker": payload.tickets[i].enable_llm_reranker,
                    "language_hint": payload.tickets[i].language_hint,
                    "display_language": payload.tickets[i].display_language,
                    "taxonomy_version": payload.tickets[i].taxonomy_version,
                }
                for i in indices
            ]
            batch_results = service.classify_batch(batch_items, provider_override=prov)
            for orig_idx, res in zip(indices, batch_results):
                ordered[orig_idx] = ClassifyTicketResponse(**asdict(res))
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return [r for r in ordered if r is not None]


@router.post("/fit")
def fit_endpoint(
    payload: FitRequest,
    container: ContainerDep,
    principal: PrincipalDep,
    response: Response,
) -> dict[str, Any]:
    """Submit a synchronous or durable asynchronous training job."""
    require_roles(principal, "trainer", "admin")
    target_domain = payload.domain_id or container.settings.domain_id
    require_domain_access(principal, target_domain)
    if payload.async_job or container.settings.auth_enabled:
        try:
            job = container.submit_fit_job(
                tenant_id=principal.tenant_id,
                domain_id=target_domain,
                payload=payload,
            )
            response.status_code = status.HTTP_202_ACCEPTED
            return FitJobResponse(
                job_id=job["job_id"],
                tenant_id=job["tenant_id"],
                domain_id=job["domain_id"],
                status=job["status"],
                idempotency_key=job["idempotency_key"],
                stage=job["stage"],
                attempts=job["attempts"],
                progress=job["progress"],
                result=job["result"],
                error=job["error"],
                created_at=job["created_at"],
                updated_at=job["updated_at"],
            ).model_dump()
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        return container.get_service(
            target_domain, tenant_id=principal.tenant_id
        ).fit_dataframe(
            pd.DataFrame(payload.records),
            strategy=payload.strategy,
            use_llm_naming=payload.use_llm_naming,
            provider_override=payload.provider_override,
            oos_texts=payload.oos_texts,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/jobs/{job_id}", response_model=FitJobResponse)
def get_fit_job_endpoint(
    job_id: str,
    container: ContainerDep,
    principal: PrincipalDep,
) -> FitJobResponse:
    require_roles(principal, "reader", "trainer", "admin")
    job = container.fit_job_repo.get_job(job_id, tenant_id=principal.tenant_id)
    if not job:
        raise HTTPException(status_code=404, detail="Fit job not found")
    require_domain_access(principal, job["domain_id"])
    return FitJobResponse(
        job_id=job["job_id"],
        tenant_id=job["tenant_id"],
        domain_id=job["domain_id"],
        status=job["status"],
        idempotency_key=job["idempotency_key"],
        stage=job["stage"],
        attempts=job["attempts"],
        progress=job["progress"],
        result=job["result"],
        error=job["error"],
        created_at=job["created_at"],
        updated_at=job["updated_at"],
    )


@router.post("/jobs/{job_id}/cancel", response_model=FitJobResponse)
def cancel_fit_job_endpoint(
    job_id: str,
    container: ContainerDep,
    principal: PrincipalDep,
) -> FitJobResponse:
    require_roles(principal, "trainer", "admin")
    job = container.fit_job_repo.get_job(job_id, tenant_id=principal.tenant_id)
    if not job:
        raise HTTPException(status_code=404, detail="Fit job not found")
    require_domain_access(principal, job["domain_id"])
    job = container.fit_job_repo.cancel_job(job_id, tenant_id=principal.tenant_id)
    return FitJobResponse(
        job_id=job["job_id"],
        tenant_id=job["tenant_id"],
        domain_id=job["domain_id"],
        status=job["status"],
        idempotency_key=job["idempotency_key"],
        stage=job["stage"],
        attempts=job["attempts"],
        progress=job["progress"],
        result=job["result"],
        error=job["error"],
        created_at=job["created_at"],
        updated_at=job["updated_at"],
    )


@router.post("/models/promote")
def promote_model_endpoint(
    payload: PromoteModelRequest,
    container: ContainerDep,
    principal: PrincipalDep,
) -> dict[str, Any]:
    require_roles(principal, "trainer", "admin")
    target_domain = payload.domain_id or container.settings.domain_id
    require_domain_access(principal, target_domain)
    if payload.approve_taxonomy:
        require_roles(principal, "admin")
    try:
        service = container.get_service(target_domain, tenant_id=principal.tenant_id)
        return service.promote_model(
            target_version=payload.target_version,
            expected_previous_version=payload.expected_previous_version,
            approved_by=principal.subject if payload.approve_taxonomy else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/models/rollback")
def rollback_model_endpoint(
    payload: RollbackModelRequest,
    container: ContainerDep,
    principal: PrincipalDep,
) -> dict[str, Any]:
    require_roles(principal, "trainer", "admin")
    target_domain = payload.domain_id or container.settings.domain_id
    require_domain_access(principal, target_domain)
    try:
        service = container.get_service(target_domain, tenant_id=principal.tenant_id)
        return service.rollback_model(target_version=payload.target_version)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/review-items")
def review_items(
    container: ContainerDep,
    principal: PrincipalDep,
    domain_id: str | None = None,
    limit: int = 100,
    cursor: str | None = None,
    item_status: str | None = None,
) -> dict[str, Any]:
    require_roles(principal, "reader", "trainer", "admin")
    if not 1 <= limit <= 1000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 1000")
    target_domain = domain_id or container.settings.domain_id
    require_domain_access(principal, target_domain)
    try:
        service = container.get_service(target_domain, tenant_id=principal.tenant_id)
        items, next_cursor = service.review_queue.list_page(
            service.profile.id,
            limit=limit,
            cursor=cursor,
            status=item_status,
            tenant_id=principal.tenant_id,
        )
        return {"items": items, "next_cursor": next_cursor}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/review-items/{ticket_id}/adjudicate")
def adjudicate_review_item_endpoint(
    ticket_id: str,
    payload: ReviewAdjudicationRequest,
    container: ContainerDep,
    principal: PrincipalDep,
) -> dict[str, Any]:
    require_roles(principal, "trainer", "admin")
    target_domain = payload.domain_id or container.settings.domain_id
    require_domain_access(principal, target_domain)
    try:
        service = container.get_service(target_domain, tenant_id=principal.tenant_id)
        return service.review_queue.adjudicate(
            domain_id=service.profile.id,
            ticket_id=ticket_id,
            status=payload.status,
            adjudicated_label=payload.adjudicated_label,
            approved_for_training=payload.approved_for_training,
            reviewer=principal.subject,
            notes=payload.notes,
            tenant_id=principal.tenant_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def create_app(container: ServiceContainer | None = None) -> FastAPI:
    """FastAPI application factory with lifespan pooling, admission limits, and readiness probes."""

    active_container = container or get_container()
    settings = active_container.settings

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            active_container.close()
            if container is None:
                get_container.cache_clear()

    app = FastAPI(
        title="Ticket and claim classification",
        docs_url="/docs" if settings.api_docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.api_docs_enabled else None,
        version="0.1.0",
        description="Discover categories and classify multilingual tickets and claims.",
        lifespan=lifespan,
    )
    if container is not None:
        app.dependency_overrides[get_container] = lambda: container

    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_bytes=lambda: (
            (container or get_container()).pipeline_config.admission.max_request_bytes
        ),
    )

    if settings.app_env == "production":
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    app.add_middleware(OperationsMiddleware)

    @app.exception_handler(Exception)
    async def internal_error(request: Request, _exc: Exception):
        logging.getLogger(__name__).error(
            "http_failure",
            extra={
                "request_id": getattr(request.state, "request_id", ""),
                "error_type": type(_exc).__name__,
            },
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal service error"},
            headers={
                "X-Request-ID": getattr(request.state, "request_id", ""),
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-store",
            },
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/health/ready")
    def readiness(response: Response, require_model: bool = False) -> dict[str, Any]:
        active_container = container or get_container()
        svc = active_container.pipeline_service
        snap = svc._capture_snapshot()
        model_ready = snap is not None
        contract_valid = (
            snap is not None
            and svc.representation_identity(
                snap.fitted_embedding_provider, profile=snap.profile
            )
            == snap.fitted_embedding_identity
        )
        require_model = require_model or settings.app_env == "production"
        ready = (not require_model or model_ready) and (
            not model_ready or contract_valid
        )
        if not ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "ready": ready,
            "model_loaded": model_ready,
            "representation_contract_valid": contract_valid,
            "active_taxonomy_version": snap.taxonomy_version if snap else None,
        }

    @app.get("/metrics")
    def metrics(
        principal: PrincipalDep,
    ) -> dict[str, Any]:
        require_roles(principal, "admin")
        if "*" not in principal.allowed_domains:
            raise HTTPException(
                status_code=403, detail="Metrics require access to all tenant domains"
            )
        active_container = container or get_container()
        svc = active_container.get_service(tenant_id=principal.tenant_id)
        return {
            **active_container._tenant_telemetry[principal.tenant_id].snapshot(),
            "job_backlog": active_container.fit_job_repo.backlog_stats(
                tenant_id=principal.tenant_id
            ),
            "review_backlog": svc.review_queue.count_backlog(
                svc.profile.id, tenant_id=principal.tenant_id
            ),
        }

    app.include_router(router)
    return app
