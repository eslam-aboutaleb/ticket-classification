"""Gateway Adapter wrapping eslam-aboutaleb/llm-gateway-sdk with switchable providers."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import TypeVar

from llm_gateway_sdk import GatewayClient
from pydantic import BaseModel

from ticket_classification.config.settings import AppSettings, LLMProvider, ModelTier

T = TypeVar("T", bound=BaseModel)


class BaseCloudGateway(ABC):
    """Abstract Strategy/Adapter interface for Cloud LLM & Embedding calls."""

    @abstractmethod
    def switch_provider(self, provider: LLMProvider | str) -> None:
        """Switch the active cloud provider ('litellm', 'openai', or 'gemini')."""

    @abstractmethod
    def get_active_provider(self) -> LLMProvider:
        """Return currently active provider."""

    @abstractmethod
    def embed_texts(
        self,
        texts: list[str],
        provider_override: LLMProvider | str | None = None,
    ) -> list[list[float]]:
        """Generate dense vector embeddings for a batch of texts."""

    @abstractmethod
    def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_cls: type[T],
        tier: ModelTier = ModelTier.STANDARD,
        provider_override: LLMProvider | str | None = None,
    ) -> T:
        """Execute a structured JSON completion validated against a Pydantic schema."""


class LLMGatewaySDKAdapter(BaseCloudGateway):
    """Concrete Gateway Adapter powered by eslam-aboutaleb/llm-gateway-sdk."""

    def __init__(
        self,
        settings: AppSettings,
        client: GatewayClient | None = None,
    ) -> None:
        self.settings = settings
        self._client = client

    def _get_client(self) -> GatewayClient:
        if self._client is not None:
            return self._client
        return self.settings.create_gateway_client()

    def switch_provider(self, provider: LLMProvider | str) -> None:
        """Dynamically switch active provider between 'litellm', 'openai', and 'gemini'."""
        self.settings.llm_provider = (
            LLMProvider(provider.lower()) if isinstance(provider, str) else provider
        )

    def get_active_provider(self) -> LLMProvider:
        return self.settings.llm_provider

    def embed_texts(
        self,
        texts: list[str],
        provider_override: LLMProvider | str | None = None,
    ) -> list[list[float]]:
        """Call /v1/embeddings on llm-gateway via llm-gateway-sdk."""
        if not texts:
            return []
        model = self.settings.resolve_model(
            ModelTier.EMBEDDING, provider_override=provider_override
        )
        dims = self.settings.embedding_dimensions
        if self._client is not None:
            resp = self._client.embed(input=texts, model=model, dimensions=dims)
            return resp.vectors
        with self.settings.create_gateway_client() as client:
            resp = client.embed(input=texts, model=model, dimensions=dims)
            return resp.vectors

    def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_cls: type[T],
        tier: ModelTier = ModelTier.STANDARD,
        provider_override: LLMProvider | str | None = None,
    ) -> T:
        """Call /v1/chat/completions on llm-gateway and parse into schema_cls."""
        model = self.settings.resolve_model(tier, provider_override=provider_override)
        schema_json = json.dumps(schema_cls.model_json_schema(), indent=2)
        full_system = (
            f"{system_prompt}\n\n"
            f"Respond ONLY with valid JSON matching this exact JSON Schema:\n{schema_json}"
        )
        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_prompt},
        ]

        if self._client is not None:
            resp = self._client.complete(
                model=model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw_content = resp.content or "{}"
        else:
            with self.settings.create_gateway_client() as client:
                resp = client.complete(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                raw_content = resp.content or "{}"

        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content.strip())
        return schema_cls.model_validate_json(cleaned)
