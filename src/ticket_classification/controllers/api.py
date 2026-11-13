"""Dependency Injection Container, FastAPI MVC Controllers, and CLI Entrypoint."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, HTTPException

from ticket_classification.config.pipeline import PipelineConfig
from ticket_classification.config.settings import AppSettings, LLMProvider
from ticket_classification.gateways.llm_gateway_adapter import (
    BaseCloudGateway,
    LLMGatewaySDKAdapter,
)
from ticket_classification.models.schemas import (
    BatchClassifyRequest,
    ClassifyTicketRequest,
    ClassifyTicketResponse,
    FitRequest,
    ProviderStatusResponse,
    ProviderSwitchRequest,
)
from ticket_classification.repositories.dataset_repo import TicketDatasetRepository
from ticket_classification.repositories.embedding_cache_repo import (
    EmbeddingCacheRepository,
)
from ticket_classification.repositories.model_registry_repo import (
    ModelRegistryRepository,
)
from ticket_classification.services.pipeline_service import TicketPipelineService
from ticket_classification.views.report_view import ReportAndPlotView


class ServiceContainer:
    """Singleton Dependency Injection container wiring Gateways, Repositories, Services, and Views."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        gateway: BaseCloudGateway | None = None,
    ) -> None:
        self.settings = settings or AppSettings()
        self.gateway = gateway or LLMGatewaySDKAdapter(self.settings)
        self._injected_gateway = gateway
        self._services: dict[str, TicketPipelineService] = {}
        self._lock = RLock()
        self.pipeline_service = self.get_service(self.settings.domain_id)
        self.dataset_repo = self.pipeline_service.dataset_repo
        self.cache_repo = self.pipeline_service.cache_repo
        self.model_registry_repo = self.pipeline_service.model_registry_repo
        self.report_view = ReportAndPlotView(
            str(Path(self.settings.data_dir) / "artifacts")
        )

    def get_service(self, domain_id: str | None = None) -> TicketPipelineService:
        domain_id = domain_id or self.settings.domain_id
        with self._lock:
            if domain_id in self._services:
                return self._services[domain_id]
            config = PipelineConfig.model_validate(self.settings.load_pipeline_yaml())
            config.profile(domain_id)  # Validate before constructing any storage paths.
            domain_settings = (
                self.settings
                if domain_id == self.settings.domain_id
                else self.settings.model_copy(
                    update={"domain_id": domain_id}, deep=True
                )
            )
            artifact_dir = Path(self.settings.data_dir) / "artifacts"
            if domain_id != "generic":
                artifact_dir = artifact_dir / domain_id
            cache_path = Path(config.embedding.cache_db_path)
            if not cache_path.is_absolute():
                cache_path = Path(self.settings.data_dir) / cache_path
            service = TicketPipelineService(
                settings=domain_settings,
                gateway=self._injected_gateway or LLMGatewaySDKAdapter(domain_settings),
                dataset_repo=TicketDatasetRepository(
                    self.settings.data_dir, config.datasets
                ),
                cache_repo=EmbeddingCacheRepository(cache_path),
                model_registry_repo=ModelRegistryRepository(
                    artifact_dir / "model_registry.joblib"
                ),
            )
            self._services[domain_id] = service
            return service


@lru_cache(maxsize=1)
def get_container() -> ServiceContainer:
    return ServiceContainer()


router = APIRouter(prefix="/api/v1")
ContainerDep = Annotated[ServiceContainer, Depends(get_container)]


@router.get("/provider", response_model=ProviderStatusResponse)
def get_provider_status(
    container: ContainerDep,
) -> ProviderStatusResponse:
    """Inspect active cloud provider ('litellm', 'openai', or 'gemini') and model tier mappings."""
    active = container.gateway.get_active_provider()
    info = container.pipeline_service.switch_provider(active)
    return ProviderStatusResponse(**info)


@router.post("/provider", response_model=ProviderStatusResponse)
def switch_provider_endpoint(
    payload: ProviderSwitchRequest,
    container: ContainerDep,
) -> ProviderStatusResponse:
    """Dynamically switch active provider between 'litellm', 'openai', and 'gemini'."""
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
    domain_id: str | None = None,
) -> dict[str, Any]:
    """Return discovered cluster profiles and taxonomy metadata."""
    try:
        service = container.get_service(domain_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return service.inspect_taxonomy()


@router.post("/classify", response_model=ClassifyTicketResponse)
def classify_ticket_endpoint(
    payload: ClassifyTicketRequest,
    container: ContainerDep,
) -> ClassifyTicketResponse:
    """Classify a single support ticket or claim against discovered clusters."""
    try:
        res = container.get_service(payload.domain_id).classify_single_ticket(
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
) -> list[ClassifyTicketResponse]:
    """Classify a batch of support tickets or claims."""
    return [
        classify_ticket_endpoint(
            item.model_copy(
                update={
                    "provider_override": payload.provider_override
                    or item.provider_override
                }
            ),
            container,
        )
        for item in payload.tickets
    ]


@router.post("/fit")
def fit_endpoint(payload: FitRequest, container: ContainerDep) -> dict[str, Any]:
    try:
        return container.get_service(payload.domain_id).fit_dataframe(
            pd.DataFrame(payload.records),
            strategy=payload.strategy,
            use_llm_naming=payload.use_llm_naming,
            provider_override=payload.provider_override,
            oos_texts=payload.oos_texts,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/review-items")
def review_items(
    container: ContainerDep, domain_id: str | None = None, limit: int = 100
) -> dict:
    if not 1 <= limit <= 1000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 1000")
    try:
        service = container.get_service(domain_id)
        return {"items": service.review_queue.list_items(service.profile.id, limit)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def create_app(container: ServiceContainer | None = None) -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title="Enterprise Text Item Clustering and Classification Service",
        version="0.1.0",
        description="Generic, cloud-native semantic clustering, taxonomy generation, and classification engine.",
    )
    if container is not None:
        app.dependency_overrides[get_container] = lambda: container

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy"}

    app.include_router(router)
    return app
