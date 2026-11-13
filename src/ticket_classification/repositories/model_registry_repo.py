"""Repository for saving and loading fitted pipeline models, centroids, encoders, and cluster taxonomies."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib


class ModelRegistryRepository:
    """Persists and restores the fitted clustering + classification pipeline state via joblib."""

    def __init__(
        self, registry_path: str | Path = "data/artifacts/model_registry.joblib"
    ) -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        """Return True if a persisted model bundle exists on disk."""
        return self.registry_path.is_file()

    def save_bundle(self, bundle: dict[str, Any]) -> str:
        """Persist the fitted pipeline bundle atomically to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        version = bundle.get("taxonomy_version")
        if version:
            versions = self.registry_path.parent / "versions"
            versions.mkdir(parents=True, exist_ok=True)
            with (versions / f"{version}.joblib").open("xb") as fh:
                joblib.dump(bundle, fh, compress=3)
        tmp_path = self.registry_path.with_suffix(f".{uuid4().hex}.tmp")
        joblib.dump(bundle, tmp_path, compress=3)
        tmp_path.replace(self.registry_path)
        return str(self.registry_path)

    def load_bundle(self) -> dict[str, Any] | None:
        """Load the persisted pipeline bundle if it exists."""
        if not self.exists():
            return None
        loaded = joblib.load(self.registry_path)
        return loaded if isinstance(loaded, dict) else None
