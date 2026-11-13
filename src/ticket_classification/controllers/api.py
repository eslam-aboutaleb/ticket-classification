"""Dependency Injection Container, FastAPI MVC Controllers, and CLI Entrypoint."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException

from ticket_classification.config.settings import AppSettings, LLMProvider
from ticket_classification.gateways.llm_gateway_adapter import (
    BaseCloudGateway,
    LLMGatewaySDKAdapter,
)
from ticket_classification.models.schemas import (
    BatchClassifyRequest,
    ClassifyTicketRequest,
    ClassifyTicketResponse,
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
        self.dataset_repo = TicketDatasetRepository(self.settings.data_dir)
        self.cache_repo = EmbeddingCacheRepository(
            f"{self.settings.data_dir}/cache/embeddings_cache.sqlite"
        )
        self.model_registry_repo = ModelRegistryRepository(
            f"{self.settings.data_dir}/artifacts/model_registry.joblib"
        )
        self.pipeline_service = TicketPipelineService(
            settings=self.settings,
            gateway=self.gateway,
            dataset_repo=self.dataset_repo,
            cache_repo=self.cache_repo,
            model_registry_repo=self.model_registry_repo,
        )
        self.report_view = ReportAndPlotView(f"{self.settings.data_dir}/artifacts")


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
) -> dict[str, Any]:
    """Return discovered cluster profiles and taxonomy metadata."""
    return {
        "cluster_count": len(container.pipeline_service.cluster_profiles),
        "clusters": [
            asdict(p) for p in container.pipeline_service.cluster_profiles.values()
        ],
    }


@router.post("/classify", response_model=ClassifyTicketResponse)
def classify_ticket_endpoint(
    payload: ClassifyTicketRequest,
    container: ContainerDep,
) -> ClassifyTicketResponse:
    """Classify a single support ticket or claim against discovered clusters."""
    try:
        res = container.pipeline_service.classify_single_ticket(
            ticket_id=payload.ticket_id,
            raw_text=payload.text,
            metadata=payload.metadata,
            provider_override=payload.provider_override,
        )
        return ClassifyTicketResponse(**asdict(res))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/classify/batch", response_model=list[ClassifyTicketResponse])
def classify_batch_endpoint(
    payload: BatchClassifyRequest,
    container: ContainerDep,
) -> list[ClassifyTicketResponse]:
    """Classify a batch of support tickets or claims."""
    results: list[ClassifyTicketResponse] = []
    for item in payload.tickets:
        res = container.pipeline_service.classify_single_ticket(
            ticket_id=item.ticket_id,
            raw_text=item.text,
            metadata=item.metadata,
            provider_override=payload.provider_override or item.provider_override,
        )
        results.append(ClassifyTicketResponse(**asdict(res)))
    return results


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
