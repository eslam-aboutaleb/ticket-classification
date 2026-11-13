"""Centralized LLM provider, model tiering, and GatewayBaseSettings configuration."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from llm_gateway_sdk.integrations.pydantic_settings import (
    GatewayBaseSettings,
    connection_options,
)
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class LLMProvider(StrEnum):
    """Switchable cloud model providers routed through llm-gateway-sdk."""

    LITELLM = (
        "litellm"  # Uses canonical gateway aliases (general-fast, rag-embedding, etc.)
    )
    OPENAI = (
        "openai"  # Uses OpenAI models (gpt-4o-mini, gpt-4o, text-embedding-3-small)
    )
    GEMINI = "gemini"  # Uses Gemini models (gemini-3.8-flash, gemini-4-argon, gemini-embedding-001)


class ModelTier(StrEnum):
    """Task tiers for routing model calls."""

    CHEAP = "cheap"  # Fast classification, stage-3 grouping, reranking
    STANDARD = "standard"  # Cluster taxonomy naming & summarization
    STRONG = "strong"  # Complex reasoning & taxonomy synthesis
    EMBEDDING = "embedding"  # Dense vector embeddings


PROVIDER_MODEL_MAP: dict[LLMProvider, dict[ModelTier, str]] = {
    LLMProvider.LITELLM: {
        ModelTier.CHEAP: "general-fast",
        ModelTier.STANDARD: "structured-output",
        ModelTier.STRONG: "general-reasoning",
        ModelTier.EMBEDDING: "rag-embedding",
    },
    LLMProvider.OPENAI: {
        ModelTier.CHEAP: "gpt-4o-mini",
        ModelTier.STANDARD: "gpt-4o-mini",
        ModelTier.STRONG: "gpt-4o",
        ModelTier.EMBEDDING: "text-embedding-3-small",
    },
    LLMProvider.GEMINI: {
        ModelTier.CHEAP: "gemini-3.8-flash",
        ModelTier.STANDARD: "gemini-3.8-flash",
        ModelTier.STRONG: "gemini-4-argon",
        ModelTier.EMBEDDING: "gemini-embedding-001",
    },
}


class AppSettings(GatewayBaseSettings):
    """Application settings inheriting secret-redacted validation from llm-gateway-sdk."""

    llm_provider: LLMProvider = Field(default=LLMProvider.LITELLM)
    llm_tiering_enabled: bool = Field(default=True)

    llm_tier_cheap_model: str = Field(default="")
    llm_tier_standard_model: str = Field(default="")
    llm_tier_strong_model: str = Field(default="")

    config_path: str = Field(default="configs/pipeline_config.yaml")
    data_dir: str = Field(default="data")
    domain_id: str = Field(default="generic", pattern=r"^[a-zA-Z0-9_-]+$")
    embedding_revision: str = ""
    embedding_task: str = "clustering"
    domain_context: str = Field(
        default="General Customer Support, Inquiries, Incidents, and Claims",
        description="Domain or industry context injected into taxonomy generation prompts",
    )
    system_role_description: str = Field(
        default="Domain Taxonomy and Triage Specialist",
        description="Persona role used for automated taxonomy labeling and reranking",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def resolve_model(
        self,
        tier: ModelTier = ModelTier.STANDARD,
        provider_override: LLMProvider | str | None = None,
    ) -> str:
        """Resolve the target model or gateway alias for the active/overridden provider."""
        provider = (
            LLMProvider(provider_override.lower())
            if isinstance(provider_override, str)
            else (provider_override or self.llm_provider)
        )
        tier_map = PROVIDER_MODEL_MAP[provider]

        if tier == ModelTier.EMBEDDING:
            if provider == LLMProvider.LITELLM and self.embedding_model:
                return self.embedding_model
            return tier_map[ModelTier.EMBEDDING]

        if self.llm_tiering_enabled:
            override = getattr(self, f"llm_tier_{tier.value}_model", "")
            if override:
                return override

        if (
            provider == LLMProvider.LITELLM
            and self.gateway_model
            and tier == ModelTier.CHEAP
        ):
            return self.gateway_model

        return tier_map[tier]

    def embedding_identity(
        self, provider_override: LLMProvider | str | None = None
    ) -> str:
        """Compute a deterministic SHA-256 fingerprint for the embedding vector space."""
        model = self.resolve_model(
            ModelTier.EMBEDDING, provider_override=provider_override
        )
        opts = connection_options(self, embedding=True)
        base = opts.get("api_base", "")
        dims = self.embedding_dimensions
        payload = json.dumps(
            [model, base, dims, self.embedding_revision, self.embedding_task],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def load_pipeline_yaml(self) -> dict[str, Any]:
        """Load pipeline hyperparameters from YAML config."""
        path = Path(self.config_path)
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
