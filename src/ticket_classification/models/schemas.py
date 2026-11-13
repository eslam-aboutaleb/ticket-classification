"""Pydantic schemas for Cloud LLM structured outputs and FastAPI REST endpoints."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ticket_classification.models.identifiers import (
    EXPECTED_VERSION_PATTERN,
    VERSION_PATTERN,
)


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

    ticket_id: str = Field(default="tkt-live-001", max_length=256)
    text: str = Field(
        max_length=32_000,
        description="Raw support ticket or claim narrative; empty inputs are queued for review",
    )
    domain_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]+$")
    taxonomy_version: str | None = Field(default=None, max_length=128)
    language_hint: str | None = Field(default=None, max_length=32)
    display_language: str | None = Field(default=None, max_length=32)
    enable_llm_reranker: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    provider_override: str | None = Field(
        default=None, description="Optional override: 'litellm', 'openai', or 'gemini'"
    )

    @field_validator("metadata")
    @classmethod
    def valid_metadata_bounds(cls, metadata: dict[str, Any]) -> dict[str, Any]:
        if len(metadata) > 64:
            raise ValueError("Metadata cannot exceed 64 keys")
        encoded = json.dumps(metadata, default=str).encode("utf-8")
        if len(encoded) > 16_384:
            raise ValueError("Metadata payload exceeds 16 KiB limit")
        return metadata


class BatchClassifyRequest(BaseModel):
    """Request payload to classify a batch of tickets or claims."""

    tickets: list[ClassifyTicketRequest] = Field(min_length=1, max_length=256)
    provider_override: str | None = None


class FitRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1, max_length=50_000)
    domain_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]+$")
    strategy: str | None = None
    provider_override: str | None = None
    use_llm_naming: bool = True
    oos_texts: list[str] | None = Field(default=None, max_length=1_000)
    idempotency_key: str | None = Field(default=None, max_length=128)
    async_job: bool = Field(
        default=False,
        description="Enqueue a durable fit job; authenticated deployments always enqueue, regardless of this flag",
    )

    @field_validator("records")
    @classmethod
    def valid_record_text_bounds(
        cls, records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        for rec in records:
            for val in rec.values():
                if isinstance(val, str) and len(val) > 32_000:
                    raise ValueError("Record text field exceeds 32,000 characters")
        return records


class FitJobResponse(BaseModel):
    """Durable fit job status and result envelope."""

    job_id: str
    tenant_id: str
    domain_id: str
    status: str
    idempotency_key: str
    stage: str = "queued"
    attempts: int = 0
    progress: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = ""
    updated_at: str = ""


class PromoteModelRequest(BaseModel):
    domain_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]+$")
    target_version: str = Field(pattern=VERSION_PATTERN)
    approve_taxonomy: bool = False
    expected_previous_version: str | None = Field(
        default=None, pattern=EXPECTED_VERSION_PATTERN
    )


class RollbackModelRequest(BaseModel):
    domain_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]+$")
    target_version: str | None = Field(
        default=None,
        max_length=128,
        pattern=VERSION_PATTERN,
        description="Version to roll back to; defaults to the release record's previous_version",
    )


class ReviewAdjudicationRequest(BaseModel):
    domain_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]+$")
    status: Literal["pending", "reviewed", "approved", "rejected", "resolved"]
    adjudicated_label: str | None = Field(default=None, max_length=256)
    approved_for_training: bool = False
    reviewer: str = Field(default="operator", max_length=128)
    notes: str = Field(default="", max_length=2000)


class ClassifyTicketResponse(BaseModel):
    """Classification response schema."""

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
