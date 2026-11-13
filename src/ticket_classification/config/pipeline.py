"""Validated runtime policies; domain and language settings compose independently."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LanguagePolicy(Policy):
    supported: list[str] = Field(default_factory=lambda: ["en", "ar"], min_length=1)
    detector: str = "langdetect"
    tokenizer: str = "unicode"
    normalization: Literal["NFC", "NFKC"] = "NFC"
    min_detection_chars: int = Field(default=12, ge=1)
    min_detection_confidence: float = Field(default=0.8, ge=0, le=1)
    lexical_plugins: dict[str, str] = Field(
        default_factory=lambda: {"en": "english", "ar": "arabic"}
    )
    taxonomy_mode: Literal["shared"] = "shared"
    display_language: str = "en"
    extra_stopwords: dict[str, list[str]] = Field(default_factory=dict)
    preserve_tokens: list[str] = Field(default_factory=list)
    arabic_remove_diacritics: bool = False
    arabic_fold_alef: bool = False


class CleaningPolicy(Policy):
    strip_email_headers: bool = True
    strip_disclaimers: bool = True
    mask_pii: bool = True
    normalize_redactions: bool = True
    min_token_length: int = Field(default=2, ge=1)
    pii_patterns: dict[str, str] = Field(default_factory=dict)
    boilerplate_patterns: list[str] = Field(default_factory=list)
    template_mask_numbers: bool = False

    @field_validator("pii_patterns", "boilerplate_patterns")
    @classmethod
    def valid_patterns(cls, patterns):
        for pattern in patterns.values() if isinstance(patterns, dict) else patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"Invalid cleaning pattern: {exc}") from exc
        return patterns


class MetadataPolicy(Policy):
    date_cols: list[str] = Field(default_factory=list)
    ordinal_cols: list[str] = Field(default_factory=list)
    nominal_cols: list[str] = Field(default_factory=list)
    numeric_cols: list[str] = Field(default_factory=list)


class DomainProfile(Policy):
    id: str = Field(default="generic", pattern=r"^[a-zA-Z0-9_-]+$")
    version: str = "1"
    context: str = "General Customer Support, Inquiries, Incidents, and Claims"
    text_col: str = "raw_text"
    id_col: str = "ticket_id"
    label_col: str | None = "label"
    language_hint_col: str | None = "language_hint"
    group_col: str | None = None
    timestamp_col: str | None = None
    split_mode: Literal["grouped", "temporal"] = "grouped"
    cross_split_duplicates: Literal["error", "prefer_holdout"] = "error"
    cleaning: CleaningPolicy = Field(default_factory=CleaningPolicy)
    language: LanguagePolicy = Field(default_factory=LanguagePolicy)
    metadata: MetadataPolicy = Field(default_factory=MetadataPolicy)
    routes: dict[str, str] = Field(
        default_factory=lambda: {"general": "General Support Queue"}
    )
    default_route: str = "general"
    review_queue: str = "Emerging Issues Triage Queue"
    chunk_chars: int = Field(default=4000, ge=100)

    @model_validator(mode="after")
    def valid_routes(self) -> DomainProfile:
        if self.default_route not in self.routes:
            raise ValueError("default_route must be an approved route ID")
        return self

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.model_dump(), sort_keys=True).encode()
        ).hexdigest()

    def semantic_fingerprint(self) -> str:
        """Fingerprint ONLY the semantic preprocessing inputs that affect contextual embeddings."""
        payload = {
            "cleaning": {
                "strip_email_headers": self.cleaning.strip_email_headers,
                "strip_disclaimers": self.cleaning.strip_disclaimers,
                "mask_pii": self.cleaning.mask_pii,
                "normalize_redactions": self.cleaning.normalize_redactions,
                "pii_patterns": self.cleaning.pii_patterns,
                "boilerplate_patterns": self.cleaning.boilerplate_patterns,
            },
            "normalization": self.language.normalization,
            "chunk_chars": self.chunk_chars,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def lexical_fingerprint(self) -> str:
        """Fingerprint lexical tokenization, stopwords, and script normalization."""
        payload = {
            "semantic": self.semantic_fingerprint(),
            "min_token_length": self.cleaning.min_token_length,
            "language": {
                "supported": sorted(self.language.supported),
                "detector": self.language.detector,
                "tokenizer": self.language.tokenizer,
                "min_detection_chars": self.language.min_detection_chars,
                "min_detection_confidence": self.language.min_detection_confidence,
                "lexical_plugins": self.language.lexical_plugins,
                "extra_stopwords": self.language.extra_stopwords,
                "preserve_tokens": sorted(self.language.preserve_tokens),
                "arabic_remove_diacritics": self.language.arabic_remove_diacritics,
                "arabic_fold_alef": self.language.arabic_fold_alef,
            },
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def presentation_fingerprint(self) -> str:
        """Fingerprint routing labels, queues, and display configuration."""
        payload = {
            "context": self.context,
            "routes": self.routes,
            "default_route": self.default_route,
            "review_queue": self.review_queue,
            "display_language": self.language.display_language,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ClusteringPolicy(Policy):
    default_strategy: str = "hdbscan_density"

    @field_validator("default_strategy")
    @classmethod
    def registered_strategy(cls, value: str) -> str:
        from ticket_classification.domain.clustering.strategies import FACTORIES

        if value not in FACTORIES:
            raise ValueError(f"Unsupported discovery strategy: {value}")
        return value

    reducer: Literal["pca", "umap", "none"] = "pca"
    n_components: int = Field(default=10, ge=1)
    k_min: int | None = Field(default=None, ge=2)
    k_max: int | None = Field(default=None, ge=2)
    candidate_budget: int = Field(default=24, ge=2)
    silhouette_sample_size: int = Field(default=1000, ge=10)
    min_cluster_support: int = Field(default=3, ge=2)
    outlier_recovery_threshold: float = Field(default=0.55, ge=0, le=1)
    recover_noise: bool = False
    single_cluster_min_similarity: float = Field(default=0.75, ge=0, le=1)
    max_discovery_samples: int = Field(default=5000, ge=50)
    max_work_budget_seconds: float = Field(default=60.0, gt=0)

    @model_validator(mode="after")
    def valid_bounds(self) -> ClusteringPolicy:
        if self.k_min and self.k_max and self.k_min > self.k_max:
            raise ValueError("k_min must not exceed k_max")
        return self


class ClassificationPolicy(Policy):
    pseudo_label_confidence_quantile: float = Field(default=0.10, ge=0, le=1)
    min_data_per_class: int = Field(default=3, ge=1)
    open_set_distance_quantile: float = Field(default=0.96, gt=0, le=1)
    min_calibration_samples: int = Field(default=8, ge=1)
    ambiguous_margin_threshold: float = Field(default=0.12, ge=0, le=1)
    enable_llm_reranker: bool = True
    lexical_weight: float = Field(default=0.25, ge=0)
    metadata_weight: float = Field(default=0.05, ge=0)
    candidate_models: list[str] = Field(
        default_factory=lambda: [
            "naive_majority",
            "logistic_regression",
            "calibrated_linear_svc",
            "gradient_boosting",
            "frozen_prototype",
        ]
    )
    export_compact_serving_bundle: bool = True


class EmbeddingPolicy(Policy):
    batch_size: int = Field(default=64, ge=1)
    max_concurrency: int = Field(default=4, ge=1)
    dtype: Literal["float32", "float64"] = "float32"
    cache_db_path: str = "cache/embeddings_cache.sqlite"
    multilingual: bool = True


class AdmissionPolicy(Policy):
    max_request_bytes: int = Field(default=2_097_152, ge=1024)
    max_text_chars: int = Field(default=32_000, ge=100)
    max_metadata_keys: int = Field(default=64, ge=1)
    max_metadata_bytes: int = Field(default=16_384, ge=128)
    max_batch_size: int = Field(default=256, ge=1)
    max_fit_records: int = Field(default=50_000, ge=1)
    max_oos_probes: int = Field(default=1_000, ge=1)
    max_concurrent_jobs_per_tenant: int = Field(default=2, ge=1)
    provider_request_budget_per_minute: int = Field(default=600, ge=1)


class ReleasePolicy(Policy):
    require_evaluation: bool = True
    require_calibration: bool = True
    min_language_samples: int = Field(default=1, ge=0)
    min_language_macro_f1: float = Field(default=0.6, ge=0, le=1)
    min_language_accepted_coverage: float = Field(default=0.5, ge=0, le=1)
    min_oos_recall: float | None = Field(default=None, ge=0, le=1)
    min_ground_truth_macro_f1: float | None = Field(default=None, ge=0, le=1)
    max_known_false_positive_rate: float | None = Field(default=None, ge=0, le=1)
    require_taxonomy_approval: bool = True


class PipelineConfig(Policy):
    project: dict[str, Any] = Field(default_factory=lambda: {"random_seed": 42})
    datasets: dict[str, dict[str, Any]] = Field(default_factory=dict)
    cleaning: CleaningPolicy = Field(default_factory=CleaningPolicy)
    embedding: EmbeddingPolicy = Field(default_factory=EmbeddingPolicy)
    clustering: ClusteringPolicy = Field(default_factory=ClusteringPolicy)
    classification: ClassificationPolicy = Field(default_factory=ClassificationPolicy)
    admission: AdmissionPolicy = Field(default_factory=AdmissionPolicy)
    release: ReleasePolicy = Field(default_factory=ReleasePolicy)
    profiles: dict[str, DomainProfile] = Field(default_factory=dict)

    def profile(self, domain_id: str) -> DomainProfile:
        builtins = {
            "generic": "General support, inquiries, incidents and claims",
            "customer_service": "Customer service and product inquiries",
            "technical_incident": "Technical incidents, error codes and service failures",
            "customer_claim": "Customer claims, disputes and compensation requests",
        }
        if domain_id in self.profiles:
            profile = self.profiles[domain_id].model_copy(deep=True)
            if profile.id != domain_id:
                raise ValueError("Profile key must match profile.id")
            return profile
        if domain_id not in builtins:
            raise ValueError(f"Unknown domain profile: {domain_id}")
        return DomainProfile(
            id=domain_id,
            context=builtins[domain_id],
            cleaning=self.cleaning.model_copy(deep=True),
        )
