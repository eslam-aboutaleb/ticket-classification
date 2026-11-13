"""Centralized LLM provider, model tiering, and GatewayBaseSettings configuration."""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import yaml
from llm_gateway_sdk.integrations.pydantic_settings import (
    GatewayBaseSettings,
    connection_options,
)
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import SettingsConfigDict


class LLMProvider(StrEnum):
    """Switchable cloud model providers routed through llm-gateway-sdk."""

    LITELLM = "litellm"
    OPENAI = "openai"
    GEMINI = "gemini"


class ModelTier(StrEnum):
    """Task tiers for routing model calls."""

    CHEAP = "cheap"
    STANDARD = "standard"
    STRONG = "strong"
    EMBEDDING = "embedding"


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
    tenant_id: str = Field(default="default", pattern=r"^[a-zA-Z0-9_-]+$")
    app_env: Literal["development", "test", "production"] = "development"
    api_docs_enabled: bool = False
    fit_job_max_attempts: int = Field(default=3, ge=1, le=10)
    allowed_hosts: list[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1"])
    auth_tokens_file: Path | None = None
    artifact_hmac_secret_file: Path | None = None
    gateway_key_file: Path | None = None
    auth_enabled: bool = Field(default=False)
    auth_tokens: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        repr=False,
        description="Optional token-to-principal map: {token: {tenant_id, roles, allowed_domains}}",
    )
    artifact_hmac_secret: str = Field(
        default="",
        repr=False,
        description="Optional HMAC secret for signed model bundle manifests",
    )
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
        hide_input_in_errors=True,
    )

    @model_validator(mode="after")
    def validate_runtime(self):
        try:
            if self.auth_tokens_file is not None:
                self.auth_tokens = json.loads(
                    self.auth_tokens_file.read_text(encoding="utf-8")
                )
            if self.artifact_hmac_secret_file is not None:
                self.artifact_hmac_secret = self.artifact_hmac_secret_file.read_text(
                    encoding="utf-8"
                ).strip()
            if self.gateway_key_file is not None:
                self.gateway_key = SecretStr(
                    self.gateway_key_file.read_text(encoding="utf-8").strip()
                )
        except (OSError, ValueError):
            raise ValueError("Unable to load configured runtime secret files") from None
        if not isinstance(self.auth_tokens, dict):
            raise ValueError(  # noqa: TRY004
                "Authentication tokens must be a token-to-principal mapping"
            )
        for token, principal in self.auth_tokens.items():
            if not isinstance(token, str) or not isinstance(principal, dict):
                raise ValueError("Invalid authentication principal configuration")  # noqa: TRY004 - Pydantic validation errors must not escape as TypeError
            tenant = principal.get("tenant_id", self.tenant_id)
            roles = principal.get("roles", ["reader"])
            domains = principal.get("allowed_domains", ["*"])
            if not isinstance(tenant, str) or not re.fullmatch(
                r"[a-zA-Z0-9_-]+", tenant
            ):
                raise ValueError("Invalid authentication tenant identifier")
            if (
                not isinstance(roles, list)
                or not roles
                or any(
                    not isinstance(role, str)
                    or role not in {"reader", "trainer", "admin"}
                    for role in roles
                )
            ):
                raise ValueError("Invalid authentication roles")
            if (
                not isinstance(domains, list)
                or not domains
                or any(
                    not isinstance(domain, str)
                    or not re.fullmatch(r"[a-zA-Z0-9_*-]+", domain)
                    or ("*" in domain and domain != "*")
                    for domain in domains
                )
            ):
                raise ValueError("Invalid authentication domain scope")
            if self.app_env == "production" and (
                len(token) < 32
                or not isinstance(principal.get("subject"), str)
                or not principal["subject"].strip()
            ):
                raise ValueError(
                    "Production tokens require at least 32 characters and an explicit subject"
                )
        if self.app_env == "production":
            if not self.auth_enabled or not self.auth_tokens:
                raise ValueError(
                    "Production requires authentication and configured principals"
                )
            if len(self.artifact_hmac_secret) < 32:
                raise ValueError(
                    "Production requires an artifact HMAC secret of at least 32 characters"
                )
            if not self.embedding_revision.strip():
                raise ValueError("Production requires an immutable embedding revision")
            if not self.allowed_hosts or "*" in self.allowed_hosts:
                raise ValueError("Production requires explicit allowed hosts")
            if not Path(self.config_path).is_file():
                raise ValueError(
                    "Production requires an existing pipeline configuration file"
                )
            if not Path(self.data_dir).is_absolute():
                raise ValueError("Production requires an absolute data directory")
            base = self.gateway_base.get_secret_value() if self.gateway_base else ""
            endpoint = urlsplit(base)
            if (
                endpoint.scheme != "https"
                or not endpoint.hostname
                or endpoint.username
                or endpoint.password
                or endpoint.query
                or endpoint.fragment
                or not self.gateway_verify_ssl
            ):
                raise ValueError(
                    "Production requires an HTTPS gateway with certificate verification"
                )
            if not self.gateway_key or not self.gateway_key.get_secret_value().strip():
                raise ValueError("Production requires a gateway API key")
            if self.gateway_max_retries != 0 or not 0 < self.gateway_timeout <= 120:
                raise ValueError(
                    "Production requires SDK retries disabled and a gateway timeout between 0 and 120 seconds"
                )
        return self

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
