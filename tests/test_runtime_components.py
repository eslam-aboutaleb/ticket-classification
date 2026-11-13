"""Focused unit tests covering repositories, API error paths, gateway retries, and lifecycle edge cases."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from llm_gateway_sdk.exceptions import GatewaySDKError

from ticket_classification.config.pipeline import DomainProfile
from ticket_classification.config.settings import AppSettings, LLMProvider
from ticket_classification.controllers.api import ServiceContainer, create_app
from ticket_classification.domain.clustering.engine import ClusteringEngine
from ticket_classification.gateways.llm_gateway_adapter import (
    BaseCloudGateway,
    LLMGatewaySDKAdapter,
)
from ticket_classification.repositories.dataset_repo import TicketDatasetRepository
from ticket_classification.repositories.fit_job_repo import FitJobRepository
from ticket_classification.repositories.model_registry_repo import (
    ModelRegistryRepository,
)
from ticket_classification.repositories.review_queue_repo import ReviewQueueRepository
from ticket_classification.services.telemetry import TelemetryCollector


class _DummyGateway(BaseCloudGateway):
    def __init__(self) -> None:
        self.provider = LLMProvider.LITELLM

    def get_active_provider(self) -> LLMProvider:
        return self.provider

    def switch_provider(self, provider: LLMProvider | str) -> None:
        self.provider = LLMProvider(provider)

    def embed_texts(
        self, texts: list[str], provider_override: LLMProvider | str | None = None
    ) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def complete_structured(self, **kwargs):
        raise ValueError("disabled")


def _make_settings(tmp_path: Path, **overrides) -> AppSettings:
    return AppSettings(
        gateway_base="https://gateway.example/v1",
        gateway_key="test-key",
        embedding_dimensions=4,
        data_dir=str(tmp_path),
        _env_file=None,
        **overrides,
    )


def test_fit_job_repo_full_lifecycle_and_checkpoints(tmp_path: Path) -> None:
    repo = FitJobRepository(tmp_path / "jobs.db")
    assert repo.backlog_stats() == {"active_jobs": 0, "oldest_age_seconds": 0.0}

    key1 = repo.compute_idempotency_key("t1", "d1", {"a": 1})
    key2 = repo.compute_idempotency_key("t1", "d1", {"a": 1}, explicit_key="custom")
    assert len(key1) == 32
    assert key2 == "custom"

    job1, is_new1 = repo.enqueue("t1", "d1", {"x": 1}, max_concurrent_jobs=1)
    assert is_new1 is True
    with pytest.raises(RuntimeError, match="Concurrent fit job limit"):
        repo.enqueue("t1", "d1", {"x": 2}, max_concurrent_jobs=1)

    stats_tenant = repo.backlog_stats(tenant_id="t1")
    stats_all = repo.backlog_stats()
    assert stats_tenant["active_jobs"] == 1
    assert stats_all["active_jobs"] == 1

    assert repo.update_stage("missing", "running", "init") is None
    assert repo.cancel_job("missing") is None

    cancelled = repo.cancel_job(job1["job_id"], tenant_id="t1")
    assert cancelled is not None and cancelled["status"] == "cancelled"
    assert repo.is_cancelled(job1["job_id"]) is True
    preserved = repo.update_stage(job1["job_id"], "running", "later")
    assert preserved is not None and preserved["status"] == "cancelled"

    requeued, is_new_re = repo.enqueue(
        "t1", "d1", {"x": 1}, idempotency_key=job1["idempotency_key"]
    )
    assert is_new_re is True
    assert requeued["job_id"] == job1["job_id"]
    assert requeued["status"] == "queued"

    completed = repo.update_stage(job1["job_id"], "completed", "done", result={"k": 2})
    assert completed is not None and completed["status"] == "completed"

    unchanged = repo.cancel_job(job1["job_id"], tenant_id="t1")
    assert unchanged is not None and unchanged["status"] == "completed"

    assert len(repo.list_jobs("t1", domain_id="d1")) == 1
    assert len(repo.list_jobs("t1")) == 1

    assert repo.load_checkpoint("t1", "d1", "embed", "cfg1") is None
    repo.save_checkpoint("t1", "d1", "embed", "cfg1", {"count": 5})
    assert repo.load_checkpoint("t1", "d1", "embed", "cfg1") == {"count": 5}


def test_api_error_paths_eviction_and_lifespan(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    gw = _DummyGateway()
    container = ServiceContainer(settings=settings, gateway=gw)
    container._max_cached_services = 1

    s1 = container.get_service("generic", tenant_id="tenant-1")
    s2 = container.get_service("generic", tenant_id="tenant-2")
    assert s1 is not s2

    with TestClient(create_app(container)) as client:
        assert client.get("/api/v1/jobs/nonexistent").status_code == 404
        assert client.post("/api/v1/jobs/nonexistent/cancel").status_code == 404
        assert (
            client.post("/api/v1/provider", json={"provider": "invalid"}).status_code
            == 400
        )
        assert (
            client.post(
                "/api/v1/classify", json={"ticket_id": "t1", "text": "hello"}
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/api/v1/classify/batch",
                json={"tickets": [{"ticket_id": "t1", "text": "hello"}]},
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/api/v1/fit",
                json={
                    "records": [{"ticket_id": "1", "raw_text": "hello"}],
                    "strategy": "nonexistent_strategy",
                },
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/api/v1/models/promote", json={"target_version": "v-missing"}
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/api/v1/models/rollback", json={"target_version": "v-missing"}
            ).status_code
            == 400
        )
        assert client.get("/api/v1/review-items?limit=0").status_code == 400
        assert (
            client.post(
                "/api/v1/review-items/missing-id/adjudicate",
                json={"status": "approved", "adjudicated_label": "billing"},
            ).status_code
            == 404
        )

        container.pipeline_config.admission.max_request_bytes = 20
        big_resp = client.post(
            "/api/v1/classify",
            json={"ticket_id": "t1", "text": "a" * 100},
        )
        assert big_resp.status_code == 413


def test_llm_gateway_adapter_retries_budget_and_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    telemetry = TelemetryCollector()
    settings = _make_settings(tmp_path)
    mock_client = MagicMock()
    monkeypatch.setattr(AppSettings, "create_gateway_client", lambda self: mock_client)

    adapter = LLMGatewaySDKAdapter(
        settings=settings,
        max_requests_per_minute=2,
        max_retries=1,
        telemetry=telemetry,
    )
    assert adapter.get_active_provider() == LLMProvider.LITELLM
    adapter.switch_provider("openai")
    assert adapter.get_active_provider() == LLMProvider.OPENAI
    assert adapter.embed_texts([]) == []

    mock_client.embed.return_value = SimpleNamespace(vectors=[[0.1, 0.2, 0.3, 0.4]])
    assert adapter.embed_texts(["hi"]) == [[0.1, 0.2, 0.3, 0.4]]

    err_400 = GatewaySDKError("bad request")
    err_400.status_code = 400  # type: ignore[attr-defined]
    mock_client.embed.side_effect = err_400
    with pytest.raises(GatewaySDKError):
        adapter.embed_texts(["hi"])
    assert telemetry.snapshot()["counters"]["provider_failures"] == 1

    with pytest.raises(RuntimeError, match="budget exceeded"):
        adapter.embed_texts(["hi"])
    assert telemetry.snapshot()["counters"]["provider_throttled"] == 1

    adapter.close()
    mock_client.close.assert_called_once()


def test_dataset_repo_adapt_and_benchmark_formats(tmp_path: Path) -> None:
    repo = TicketDatasetRepository(tmp_path)
    profile = DomainProfile(
        id="custom",
        text_col="body",
        id_col="id",
        label_col="lbl",
        language_hint_col="lang",
    )

    with pytest.raises(ValueError, match="Missing configured text column"):
        repo.adapt_frame(pd.DataFrame({"wrong": ["a"]}), profile)

    with pytest.raises(ValueError, match="non-null and unique"):
        repo.adapt_frame(pd.DataFrame({"id": ["1", "1"], "body": ["a", "b"]}), profile)

    adapted = repo.adapt_frame(
        pd.DataFrame(
            {
                "id": ["1", "2"],
                "body": ["hello", "world"],
                "lbl": ["a", "b"],
                "lang": ["en", "en"],
                "metadata": [{"region": "us"}, {"region": "eu"}],
            }
        ),
        profile,
    )
    assert list(adapted["label"]) == ["a", "b"]
    assert list(adapted["language_hint"]) == ["en", "en"]
    assert list(adapted["region"]) == ["us", "eu"]

    with pytest.raises(ValueError, match="Unknown dataset"):
        repo.load_benchmark("does_not_exist")

    csv_path = repo.raw_dir / "it_service_tickets.csv"
    csv_path.write_text(
        "Document,Topic_group\nCharged twice,Credit card\nWrong fee,Checking\n",
        encoding="utf-8",
    )
    df_csv = repo.load_benchmark("it_service_tickets")
    assert len(df_csv) == 2
    assert set(df_csv["label"]) == {"Credit card", "Checking"}

    aw_path = repo.raw_dir / "airweave_errors.py"
    aw_path.write_text(
        'SAMPLES = [{"ts": datetime.now(), "message": "Timeout on sync"}]\n',
        encoding="utf-8",
    )
    df_aw = repo.load_benchmark("airweave_incidents")
    assert len(df_aw) == 1
    assert df_aw.iloc[0]["raw_text"] == "Timeout on sync"


def test_review_queue_and_model_registry_edge_cases(tmp_path: Path) -> None:
    rq = ReviewQueueRepository(tmp_path / "reviews.db")
    rq.put(
        domain_id="generic",
        ticket_id="t-1",
        version="v1",
        text="low conf",
        languages=["en"],
        reason="low_confidence",
    )
    assert rq.count_backlog("generic") == 1
    assert len(rq.list_items("generic", status="pending")) == 1

    with pytest.raises(ValueError, match="approved_for_training"):
        rq.adjudicate(
            domain_id="generic",
            ticket_id="t-1",
            status="pending",
            adjudicated_label="billing",
            approved_for_training=True,
        )

    rq.adjudicate(
        domain_id="generic",
        ticket_id="t-1",
        status="approved",
        adjudicated_label="billing",
        approved_for_training=True,
        reviewer="alice",
    )
    assert rq.count_backlog("generic") == 0
    assert len(rq.list_approved_training_items("generic")) == 1
    assert len(rq.list_audit_events("generic", ticket_id="t-1")) == 2
    assert len(rq.list_audit_events("generic")) == 2

    reg = ModelRegistryRepository(tmp_path / "registry.joblib", hmac_secret="secret")
    bundle = {"taxonomy_version": "v1", "domain_id": "generic", "payload": [1, 2, 3]}
    reg.save_bundle(bundle)
    assert reg.get_release_record() is not None
    assert reg.get_release_record()["active_version"] == "v1"

    with pytest.raises(ValueError, match="expected_previous_version|conflict"):
        reg.promote_version("v1", expected_previous_version="wrong-prev")
    with pytest.raises(ValueError, match="not found|does not exist"):
        reg.rollback("v-missing")

    manifest_file = reg.versions_dir / "v1.joblib.manifest.json"
    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest_data["hmac_sha256"] = "0" * 64
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")
    with pytest.raises(ValueError, match="signature verification failed"):
        reg.load_bundle()


def test_clustering_engine_cancel_and_unfitted_transform() -> None:
    engine = ClusteringEngine(
        strategy="aligned_kmeans", reducer="none", k_min=2, k_max=3
    )
    with pytest.raises(RuntimeError, match="called before fit"):
        engine.transform(np.ones((2, 4)))

    cancelled_engine = ClusteringEngine(
        strategy="aligned_kmeans",
        reducer="none",
        k_min=2,
        k_max=3,
        cancel_check=lambda: True,
    )
    rng = np.random.default_rng(0)
    data = rng.normal(size=(12, 4))
    with pytest.raises(RuntimeError, match="cancelled"):
        cancelled_engine.fit_transform(data)
