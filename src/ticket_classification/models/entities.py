"""Domain entities and value objects for the MVC Model layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TicketRecord:
    """Represents a single support ticket or customer claim."""

    ticket_id: str
    raw_text: str
    ground_truth_label: str | None = None
    split: str = "train"  # train | val | test
    metadata: dict[str, Any] = field(default_factory=dict)
    missing_indicators: dict[str, int] = field(default_factory=dict)
    text_contextual: str = ""
    text_lexical: str = ""
    template_signature: str = ""
    cluster_id: int | None = None
    cluster_confidence: float = 0.0
    languages: list[str] = field(default_factory=list)
    language_confidence: float = 0.0
    language_status: str = "undetermined"
    language_hint: str | None = None


@dataclass
class ClusterProfile:
    """Represents a discovered semantic cluster and its taxonomy metadata."""

    cluster_id: int
    size: int
    top_keywords: list[str]
    medoid_texts: list[str]
    short_label: str
    signature: str
    routing_queue: str
    centroid: list[float] = field(default_factory=list)
    stable_id: str = ""
    route_id: str = "general"
    keywords_by_language: dict[str, list[str]] = field(default_factory=dict)
    localized_labels: dict[str, str] = field(default_factory=dict)


@dataclass
class AlignmentResult:
    """Result of Hungarian centroid alignment and outlier recovery."""

    num_clusters: int
    silhouette_score: float
    recovered_outliers: int
    remaining_outliers: int
    alignment_map: dict[int, int]


@dataclass
class PredictionResult:
    """Result of classifying an incoming ticket against discovered clusters."""

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
    top_contributing_terms: list[str] = field(default_factory=list)
    reasoning: str = ""
    status: str = "known"
    domain_id: str = "generic"
    taxonomy_version: str = ""
    stable_cluster_id: str | None = None
    route_id: str | None = None
    input_languages: list[str] = field(default_factory=list)
    language_confidence: float = 0.0
    language_status: str = "undetermined"
    language_hint: str | None = None
    display_language: str = "en"
