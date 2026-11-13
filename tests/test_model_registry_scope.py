"""Tenant-scoped persistence, reload and recovery from pre-provenance artifacts."""

import json
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd
import pytest
from test_production_readiness import ControlledGateway, _make_settings, _sample_records

from ticket_classification.controllers.api import ServiceContainer
from ticket_classification.repositories.model_registry_repo import (
    LegacyArtifactError,
    ModelRegistryRepository,
)
from ticket_classification.services.pipeline_service import TicketPipelineService


def test_tenant_models_reload_through_container_and_direct_service(tmp_path):
    settings = _make_settings(
        tmp_path, tenant_id="tenant-a", artifact_hmac_secret="test-signing-key"
    )
    gateway = ControlledGateway()
    first = ServiceContainer(settings, gateway)
    versions = {}
    try:
        for tenant in ("tenant-a", "tenant-b"):
            service = first.get_service("customer_claim", tenant_id=tenant)
            metrics = service.fit_dataframe(
                pd.DataFrame(_sample_records(10)), use_llm_naming=False
            )
            versions[tenant] = metrics["taxonomy_version"]
            manifest = service.model_registry_repo.version_path(
                versions[tenant]
            ).with_suffix(".joblib.manifest.json")
            assert json.loads(manifest.read_text())["metadata"]["tenant_id"] == tenant
    finally:
        first.close()
    second = ServiceContainer(settings, ControlledGateway())
    try:
        for tenant, version in versions.items():
            service = second.get_service("customer_claim", tenant_id=tenant)
            assert service._capture_snapshot().tenant_id == tenant
            assert service._capture_snapshot().taxonomy_version == version
            direct = TicketPipelineService(
                service.settings,
                service.gateway,
                service.dataset_repo,
                service.cache_repo,
            )
            assert (
                direct.model_registry_repo.registry_path
                == service.model_registry_repo.registry_path
            )
            assert direct._capture_snapshot().taxonomy_version == version
        assert len(set(versions.values())) == 2
    finally:
        second.close()


def test_legacy_manifest_never_deserializes_but_allows_refit_and_reload(tmp_path):
    settings = _make_settings(tmp_path, artifact_hmac_secret="test-signing-key")
    container = ServiceContainer(settings, ControlledGateway())
    metrics = container.pipeline_service.fit_dataframe(
        pd.DataFrame(_sample_records(10)), use_llm_naming=False
    )
    registry = container.model_registry_repo
    old_version = metrics["taxonomy_version"]
    artifact = registry.version_path(old_version)
    artifact_bytes = artifact.read_bytes()
    manifest_path = artifact.with_suffix(".joblib.manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"].pop("tenant_id")
    manifest["hmac_sha256"] = registry._compute_hmac(
        manifest["sha256"], manifest["metadata"]
    )
    manifest_path.write_text(json.dumps(manifest))
    container.close()
    real_load = joblib.load

    def guarded_load(path, *args, **kwargs):
        assert Path(path) != artifact, "Legacy artifact must not be deserialized"
        return real_load(path, *args, **kwargs)

    with patch(
        "ticket_classification.repositories.model_registry_repo.joblib.load",
        side_effect=guarded_load,
    ):
        recovered = ServiceContainer(settings, ControlledGateway())
        try:
            assert recovered.pipeline_service._capture_snapshot() is None
            with pytest.raises(LegacyArtifactError, match="lacks tenant provenance"):
                registry.load_bundle()
            rebuilt = recovered.pipeline_service.fit_dataframe(
                pd.DataFrame(_sample_records(10)), use_llm_naming=False
            )
            assert rebuilt["publication"]["status"] == "active"
            assert (
                recovered.model_registry_repo.get_release_record()["previous_version"]
                == old_version
            )
            assert artifact.read_bytes() == artifact_bytes
        finally:
            recovered.close()
        restarted = ServiceContainer(settings, ControlledGateway())
        try:
            assert (
                restarted.pipeline_service._capture_snapshot().taxonomy_version
                == rebuilt["taxonomy_version"]
            )
        finally:
            restarted.close()


def test_foreign_tenant_is_not_treated_as_legacy(tmp_path):
    settings = _make_settings(tmp_path)
    registry = ModelRegistryRepository.for_scope(
        tmp_path, tenant_id="foreign", enforce_release_gate=False
    )
    registry.save_bundle(
        {
            "taxonomy_version": "foreign-version",
            "tenant_id": "foreign",
            "domain_id": "generic",
        }
    )
    wrong = ModelRegistryRepository(
        registry.registry_path, tenant_id="default", domain_id="generic"
    )
    with pytest.raises(ValueError, match="different tenant"):
        TicketPipelineService(
            settings, ControlledGateway(), None, None, model_registry_repo=wrong
        )


def test_registry_scope_paths_are_consistent_and_distinct(tmp_path):
    paths = {
        ModelRegistryRepository.for_scope(tmp_path, tenant, domain).registry_path
        for tenant in ("default", "tenant-a")
        for domain in ("generic", "customer_claim")
    }
    assert len(paths) == 4
    for tenant, domain in [("../tenant-b", "generic"), ("tenant-a", "../generic")]:
        with pytest.raises(ValueError, match="Invalid registry scope"):
            ModelRegistryRepository.for_scope(tmp_path, tenant, domain)
