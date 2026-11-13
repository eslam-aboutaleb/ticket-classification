"""Gateway Adapter wrapping eslam-aboutaleb/llm-gateway-sdk with switchable providers, persistent pooling, and bounded concurrency."""

from __future__ import annotations

import json
import math
import random
import re
import time
from abc import ABC, abstractmethod
from collections import deque
from threading import RLock, Semaphore
from typing import TypeVar

from llm_gateway_sdk import GatewayClient
from llm_gateway_sdk.exceptions import GatewaySDKError
from pydantic import BaseModel

from ticket_classification.config.settings import AppSettings, LLMProvider, ModelTier
from ticket_classification.services.telemetry import TelemetryCollector

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

    def close(self) -> None:
        """Release any pooled network connections."""


class LLMGatewaySDKAdapter(BaseCloudGateway):
    """Concrete Gateway Adapter powered by eslam-aboutaleb/llm-gateway-sdk with connection reuse and bounded concurrency."""

    def __init__(
        self,
        settings: AppSettings,
        client: GatewayClient | None = None,
        max_concurrency: int = 4,
        max_requests_per_minute: int = 600,
        max_retries: int = 2,
        telemetry: TelemetryCollector | None = None,
    ) -> None:
        self.settings = settings
        self._client = client
        self._owns_client = False
        self._lock = RLock()
        self._semaphore = Semaphore(max(1, max_concurrency))
        self.max_requests_per_minute = max(1, max_requests_per_minute)
        self.max_retries = max(0, max_retries)
        self._request_timestamps: deque[float] = deque()
        self.telemetry = telemetry

    def _get_client(self) -> GatewayClient:
        with self._lock:
            if self._client is None:
                self._client = self.settings.create_gateway_client()
                self._owns_client = True
            return self._client

    def close(self) -> None:
        with self._lock:
            if self._owns_client and self._client is not None:
                close_fn = getattr(self._client, "close", None)
                if callable(close_fn):
                    close_fn()
                self._client = None
                self._owns_client = False

    def _check_budget(self) -> None:
        now = time.monotonic()
        with self._lock:
            while (
                self._request_timestamps and now - self._request_timestamps[0] >= 60.0
            ):
                self._request_timestamps.popleft()
            if len(self._request_timestamps) >= self.max_requests_per_minute:
                if self.telemetry:
                    self.telemetry.inc("provider_throttled")
                raise RuntimeError("Provider request budget exceeded; retry later")
            self._request_timestamps.append(now)

    def _call_with_retry(self, fn):
        with self._semaphore:
            attempt = 0
            while True:
                self._check_budget()
                try:
                    return fn()
                except GatewaySDKError as exc:
                    status_code = getattr(exc, "status_code", None)
                    transient = status_code in {None, 408, 429, 500, 502, 503, 504}
                    if attempt >= self.max_retries or not transient:
                        if self.telemetry:
                            self.telemetry.inc("provider_failures")
                        raise
                    attempt += 1
                    if self.telemetry:
                        self.telemetry.inc("provider_retries")
                    retry_after = getattr(exc, "retry_after", None)
                    delay = (
                        float(retry_after)
                        if retry_after is not None
                        else min(2.0, (0.05 * (2**attempt)) + random.uniform(0.0, 0.05))
                    )
                    time.sleep(
                        min(10.0, max(0.0, delay)) if math.isfinite(delay) else 2.0
                    )

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
        client = self._get_client()
        resp = self._call_with_retry(
            lambda: client.embed(input=texts, model=model, dimensions=dims)
        )
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

        client = self._get_client()
        resp = self._call_with_retry(
            lambda: client.complete(
                model=model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        )
        raw_content = resp.content or "{}"
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content.strip())
        return schema_cls.model_validate_json(cleaned)
