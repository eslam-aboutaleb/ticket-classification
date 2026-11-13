"""Pydantic schemas for Cloud LLM structured outputs and FastAPI REST endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ClusterTaxonomyLLMOutput(BaseModel):
    """Structured output schema for Cloud LLM cluster naming and routing mapping."""

    short_label: str = Field(description="Concise category title (2 to 5 words)")
    signature: str = Field(
        description="Natural language signature (50 to 150 chars) summarizing the issue pattern"
    )
    routing_queue: str = Field(
        description="Recommended support department or claims routing queue"
    )
    localized_labels: dict[str, str] = Field(
        default_factory=dict,
        description="Optional labels keyed by language code; category identity stays unchanged",
    )


class RerankDecisionLLMOutput(BaseModel):
    """Structured output schema when the Cloud LLM resolves an ambiguous boundary ticket."""

    chosen_cluster_id: int = Field(
        description="Best matching cluster_id from candidates, or -1 if brand new unseen issue"
    )
    confidence: float = Field(
        ge=0, le=1, description="Confidence score between 0.0 and 1.0"
    )
    is_novel_issue: bool = Field(
        description="True if ticket does not fit any candidate cluster"
    )
    reasoning: str = Field(
        description="Brief explanation of why this cluster was selected"
    )


class ProviderSwitchRequest(BaseModel):
    """Request payload to switch the active Cloud LLM provider."""

    provider: str = Field(
        description="Target provider: 'litellm', 'openai', or 'gemini'"
    )


class ProviderStatusResponse(BaseModel):
    """Response showing the active provider and resolved models per tier."""

    active_provider: str
    gateway_base_url: str
    models_by_tier: dict[str, str]
    embedding_identity: str


class ClassifyTicketRequest(BaseModel):
    """Request payload to classify a single ticket or claim."""

    ticket_id: str = Field(default="tkt-live-001")
    text: str = Field(
        description="Raw support ticket or claim narrative; empty inputs are queued for review"
    )
    domain_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]+$")
    taxonomy_version: str | None = None
    language_hint: str | None = None
    display_language: str | None = None
    enable_llm_reranker: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    provider_override: str | None = Field(
        default=None, description="Optional override: 'litellm', 'openai', or 'gemini'"
    )


class BatchClassifyRequest(BaseModel):
    """Request payload to classify a batch of tickets or claims."""

    tickets: list[ClassifyTicketRequest]
    provider_override: str | None = None


class FitRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1)
    domain_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]+$")
    strategy: str | None = None
    provider_override: str | None = None
    use_llm_naming: bool = True
    oos_texts: list[str] | None = None


class ClassifyTicketResponse(BaseModel):
    """Formatted classification response presented by the MVC View layer."""

    ticket_id: str
    predicted_cluster_id: int
    cluster_label: str
    routing_queue: str
    confidence: float
    margin: float
    is_open_set_novel: bool
    centroid_distance: float
    used_llm_reranker: bool
    provider_used: str
    top_contributing_terms: list[str]
    reasoning: str
    status: str
    domain_id: str
    taxonomy_version: str
    stable_cluster_id: str | None
    route_id: str | None
    input_languages: list[str]
    language_confidence: float
    language_status: str
    language_hint: str | None
    display_language: str
