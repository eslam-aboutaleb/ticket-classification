"""Discovery strategy registration for application-supplied implementations."""

from collections.abc import Callable
from functools import partial
from typing import Any, Protocol

import numpy as np

from ticket_classification.domain.clustering.engine import ClusteringEngine


class DiscoveryStrategy(Protocol):
    optimal_k_: int
    diagnostics_: dict[str, Any]
    raw_space_centroids_: np.ndarray

    def fit_transform(
        self, embeddings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...
    def transform(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


FACTORIES: dict[str, Callable[..., ClusteringEngine]] = {
    name: partial(ClusteringEngine, strategy=name)
    for name in ("aligned_kmeans", "hdbscan_density", "density")
}


def register_discovery_strategy(
    name: str, factory: Callable[..., ClusteringEngine]
) -> None:
    """Register a compatible engine factory at startup, before config validation."""
    if name in FACTORIES:
        raise ValueError(f"Discovery strategy already registered: {name}")
    FACTORIES[name] = factory


def create_discovery_strategy(name: str, **options) -> ClusteringEngine:
    if name not in FACTORIES:
        raise ValueError(f"Unsupported discovery strategy: {name}")
    return FACTORIES[name](**options)
