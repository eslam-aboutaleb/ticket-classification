"""Tenant-scoped immutable artifacts and serialized single-host model promotion."""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import re
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import sklearn

from ticket_classification.models.identifiers import VERSION_PATTERN


def runtime_environment_revisions() -> dict[str, str]:
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


class LegacyArtifactError(ValueError):
    """The artifact predates tenant provenance and must be refitted, never inferred."""


class ModelRegistryRepository:
    """Use a shared local filesystem on one host; flock is not a multi-host registry."""

    def __init__(
        self,
        registry_path: str | Path = "data/artifacts/model_registry.joblib",
        hmac_secret: str = "",
        tenant_id: str = "default",
        domain_id: str | None = None,
        enforce_release_gate: bool = False,
    ) -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.hmac_secret = hmac_secret
        self.tenant_id = tenant_id
        self.domain_id = domain_id
        self.enforce_release_gate = enforce_release_gate

    @classmethod
    def for_scope(
        cls,
        data_dir: str | Path,
        tenant_id: str = "default",
        domain_id: str = "generic",
        hmac_secret: str = "",
        enforce_release_gate: bool = True,
    ) -> ModelRegistryRepository:
        for identifier in (tenant_id, domain_id):
            if re.fullmatch(r"[a-zA-Z0-9_-]+", identifier) is None:
                raise ValueError("Invalid registry scope identifier")
        root = Path(data_dir) / "artifacts"
        if tenant_id != "default":
            root = root / "tenants" / tenant_id
        if domain_id != "generic":
            root = root / domain_id
        return cls(
            root / "model_registry.joblib",
            hmac_secret,
            tenant_id,
            domain_id,
            enforce_release_gate,
        )

    @property
    def versions_dir(self) -> Path:
        return self.registry_path.parent / "versions"

    @property
    def release_record_path(self) -> Path:
        return self.registry_path.parent / "release_record.json"

    @contextmanager
    def _publication_lock(self):
        with self.registry_path.with_suffix(".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def version_path(self, version: str, *, research: bool = False) -> Path:
        if (
            not isinstance(version, str)
            or re.fullmatch(VERSION_PATTERN, version) is None
        ):
            raise ValueError("Invalid model version identifier")
        root = self.versions_dir.resolve()
        suffix = ".research.joblib" if research else ".joblib"
        path = root / f"{version}{suffix}"
        # Reject symlinks, including links back inside the registry: versions are immutable files.
        if path.is_symlink() or path.resolve().parent != root:
            raise ValueError("Model artifact escapes its scoped registry")
        return path

    def exists(self) -> bool:
        rec = self.get_release_record()
        if rec:
            return self.version_path(rec["active_version"]).is_file()
        return self.registry_path.is_file()

    @staticmethod
    def _compute_file_digest(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as fh:
            while chunk := fh.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _compute_hmac(self, digest: str, metadata: dict[str, Any]) -> str:
        if not self.hmac_secret:
            return ""
        canonical = json.dumps(
            {"sha256": digest, "metadata": metadata}, sort_keys=True
        ).encode()
        return hmac.new(
            self.hmac_secret.encode(), canonical, hashlib.sha256
        ).hexdigest()

    def _write_manifest(
        self, artifact_path: Path, bundle: dict[str, Any]
    ) -> dict[str, Any]:
        digest = self._compute_file_digest(artifact_path)
        profile = bundle.get("profile")
        meta = {
            "format_version": int(bundle.get("format_version", 2)),
            "tenant_id": bundle.get("tenant_id", "default"),
            "domain_id": str(
                getattr(profile, "id", None) or bundle.get("domain_id", "generic")
            ),
            "taxonomy_version": str(bundle.get("taxonomy_version", "")),
            "fitted_embedding_identity": bundle.get("fitted_embedding_identity"),
            "fitted_embedding_width": bundle.get("fitted_embedding_width"),
            "release_gate": bundle.get("release_gate"),
            "environment": runtime_environment_revisions(),
        }
        manifest = {
            "sha256": digest,
            "metadata": meta,
            "hmac_sha256": self._compute_hmac(digest, meta),
            "created_at": datetime.now(UTC).isoformat(),
        }
        self._atomic_json(
            artifact_path.with_suffix(artifact_path.suffix + ".manifest.json"), manifest
        )
        return manifest

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, Any]) -> None:
        tmp = path.with_suffix(f".{uuid4().hex}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(value, fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)

    def verify_artifact_before_load(
        self,
        artifact_path: Path,
        expected_domain_id: str | None = None,
        require_manifest: bool = False,
    ) -> dict[str, Any] | None:
        manifest_path = artifact_path.with_suffix(
            artifact_path.suffix + ".manifest.json"
        )
        if not manifest_path.is_file():
            if (
                require_manifest
                or self.hmac_secret
                or self.domain_id
                or self.tenant_id != "default"
            ):
                raise ValueError(
                    f"Untrusted model artifact '{artifact_path.name}': missing signed manifest"
                )
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ValueError(f"Malformed artifact manifest: {exc}") from exc
        actual_digest = self._compute_file_digest(artifact_path)
        if actual_digest != manifest.get("sha256"):
            raise ValueError(
                f"Artifact integrity check failed for '{artifact_path.name}': SHA-256 mismatch"
            )
        meta = manifest.get("metadata", {})
        if self.hmac_secret and not hmac.compare_digest(
            self._compute_hmac(actual_digest, meta),
            str(manifest.get("hmac_sha256", "")),
        ):
            raise ValueError(
                f"Artifact signature verification failed for '{artifact_path.name}'"
            )
        if int(meta.get("format_version", 0)) != 2:
            raise ValueError(
                f"Incompatible artifact format_version: {meta.get('format_version')}"
            )
        if meta.get("tenant_id") is None:
            raise LegacyArtifactError(
                "Model artifact lacks tenant provenance; refit before serving"
            )
        if meta.get("tenant_id") != self.tenant_id:
            raise ValueError("Model bundle belongs to a different tenant")
        domain = expected_domain_id or self.domain_id
        if domain and meta.get("domain_id") != domain:
            raise ValueError("Model bundle belongs to a different domain")
        return manifest

    def save_bundle(
        self,
        bundle: dict[str, Any],
        compact_bundle: dict[str, Any] | None = None,
        expected_previous_version: str | None = None,
        *,
        promote: bool = True,
    ) -> str:
        serving = compact_bundle if compact_bundle is not None else bundle
        for payload in (bundle, serving):
            if payload.get("tenant_id", "default") != self.tenant_id:
                raise ValueError("Model bundle belongs to a different tenant")
            domain = getattr(payload.get("profile"), "id", None) or payload.get(
                "domain_id", "generic"
            )
            if self.domain_id and domain != self.domain_id:
                raise ValueError("Model bundle belongs to a different domain")
            if payload.get("taxonomy_version") != bundle.get("taxonomy_version"):
                raise ValueError("Serving and research versions must match")
        version = bundle.get("taxonomy_version")
        if version is not None:
            self.version_path(version)
            self.versions_dir.mkdir(parents=True, exist_ok=True)
            # Writers serialize candidate creation as well as pointer updates.
            with self._publication_lock():
                for payload, research in [(serving, False)] + (
                    [(bundle, True)] if compact_bundle is not None else []
                ):
                    target = self.version_path(version, research=research)
                    if target.exists():
                        raise FileExistsError(
                            f"Immutable version '{version}' already exists"
                        )
                    tmp = target.with_suffix(f".{uuid4().hex}.tmp")
                    try:
                        joblib.dump(payload, tmp, compress=3)
                        self._write_manifest(tmp, payload)
                        tmp_manifest = tmp.with_suffix(tmp.suffix + ".manifest.json")
                        tmp_manifest.replace(
                            target.with_suffix(target.suffix + ".manifest.json")
                        )
                        tmp.replace(target)
                    finally:
                        tmp.unlink(missing_ok=True)
            if promote:
                self.promote_version(version, expected_previous_version)
            return str(self.version_path(version))
        if self.enforce_release_gate:
            raise ValueError("A versioned candidate is required")
        with self._publication_lock():
            joblib.dump(serving, self.registry_path, compress=3)
            self._write_manifest(self.registry_path, serving)
        return str(self.registry_path)

    def get_release_record(self) -> dict[str, Any] | None:
        if not self.release_record_path.is_file():
            return None
        # A corrupt release pointer must fail closed, never fall back to another artifact.
        return json.loads(self.release_record_path.read_text(encoding="utf-8"))

    def promote_version(
        self,
        target_version: str,
        expected_previous_version: str | None = None,
        *,
        approved_by: str | None = None,
    ) -> dict[str, Any]:
        with self._publication_lock():
            version_file = self.version_path(target_version)
            if not version_file.is_file():
                raise ValueError(f"Target version '{target_version}' does not exist")
            manifest = self.verify_artifact_before_load(
                version_file, require_manifest=True
            )
            meta = manifest["metadata"]
            if meta.get("taxonomy_version") != target_version:
                raise ValueError("Artifact version does not match requested version")
            current = self.get_release_record() or {}
            active = current.get("active_version")
            if (
                expected_previous_version is not None
                and (active or "") != expected_previous_version
            ):
                raise ValueError(
                    f"CAS promotion conflict: expected active version '{expected_previous_version}', found '{active}'"
                )
            approvals = dict(current.get("approvals", {}))
            if self.enforce_release_gate:
                gate = meta.get("release_gate") or {}
                if gate.get("passed") is not True:
                    raise ValueError(
                        f"Candidate failed release checks: {gate.get('blockers', ['missing release evidence'])}"
                    )
                if gate.get("requires_human_taxonomy_review") and not (
                    approved_by or target_version in approvals
                ):
                    raise ValueError(
                        "Candidate requires explicit human taxonomy approval"
                    )
                if approved_by:
                    approvals[target_version] = {
                        "reviewer": approved_by,
                        "approved_at": datetime.now(UTC).isoformat(),
                    }
            history = list(current.get("history", []))
            if active and active != target_version:
                history.append(active)
            record = {
                "active_version": target_version,
                "previous_version": active
                if active != target_version
                else current.get("previous_version"),
                "history": history[-20:],
                "approvals": approvals,
                "promoted_at": datetime.now(UTC).isoformat(),
            }
            # Only the release pointer is authoritative; candidates never overwrite a fallback active file.
            self._atomic_json(self.release_record_path, record)
            return record

    def rollback(self, target_version: str | None = None) -> dict[str, Any]:
        current = self.get_release_record()
        if not current:
            raise ValueError("No release record available to roll back")
        target = target_version or current.get("previous_version")
        if not target:
            raise ValueError("No previous model version recorded for rollback")
        return self.promote_version(str(target), current.get("active_version"))

    def load_bundle(
        self,
        version: str | None = None,
        expected_domain_id: str | None = None,
        require_manifest: bool = False,
    ) -> dict[str, Any] | None:
        if version is None:
            rec = self.get_release_record()
            version = rec.get("active_version") if rec else None
        target = (
            self.version_path(version) if version is not None else self.registry_path
        )
        if not target.is_file():
            if version:
                raise ValueError("Active or requested model artifact is missing")
            return None
        self.verify_artifact_before_load(target, expected_domain_id, require_manifest)
        loaded = joblib.load(target)
        if (
            isinstance(loaded, dict)
            and loaded.get("tenant_id", "default") != self.tenant_id
        ):
            raise ValueError("Model bundle belongs to a different tenant")
        return loaded if isinstance(loaded, dict) else None
