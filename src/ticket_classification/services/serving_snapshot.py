"""Immutable serving snapshot for concurrent, lock-free online inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

from ticket_classification.config.pipeline import DomainProfile, PipelineConfig
from ticket_classification.config.settings import LLMProvider
from ticket_classification.domain.classification.engine import ClassifierSuite
from ticket_classification.domain.cleaning.handlers import TextCleaningHandler
from ticket_classification.domain.clustering.engine import ClusteringEngine
from ticket_classification.domain.features.metadata_encoder import (
    MetadataFeaturePipeline,
)
from ticket_classification.models.entities import ClusterProfile


@dataclass(frozen=True)
class ServingSnapshot:
    """Validated model state captured for one inference request."""

    tenant_id: str
    domain_id: str
    taxonomy_version: str
    profile: DomainProfile
    config: PipelineConfig
    cleaning_chain: TextCleaningHandler
    metadata_pipeline: MetadataFeaturePipeline
    tfidf_vectorizer: TfidfVectorizer
    clustering_engine: ClusteringEngine
    classifier_suite: ClassifierSuite
    cluster_profiles: dict[int, ClusterProfile]
    stable_cluster_ids: dict[int, str]
    lineage: dict[str, Any]
    fitted_embedding_identity: str
    fitted_embedding_width: int
    fitted_embedding_provider: LLMProvider | None
    sparse_features: bool = False
