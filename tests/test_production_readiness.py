"""Authorization, concurrency, batching and durable storage behavior."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from threading import Thread
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import yaml
from conftest import SYNTHETIC_RELEASE_POLICY
from fastapi.testclient import TestClient
from llm_gateway_sdk.exceptions import GatewaySDKError

from ticket_classification.config.settings import AppSettings, LLMProvider
from ticket_classification.controllers.api import ServiceContainer, create_app
from ticket_classification.domain.clustering.engine import ClusteringEngine
from ticket_classification.gateways.llm_gateway_adapter import (
    BaseCloudGateway,
    LLMGatewaySDKAdapter,
)
from ticket_classification.repositories.model_registry_repo import (
    ModelRegistryRepository,
)
from ticket_classification.services.fit_worker import FitWorker


class ControlledGateway(BaseCloudGateway):
    def __init__(
        self, fail_on_batch: int | None = None, slow_seconds: float = 0.0
    ) -> None:
        self.provider = LLMProvider.LITELLM
        self.embed_calls: list[list[str]] = []
        self.fail_on_batch = fail_on_batch
        self.slow_seconds = slow_seconds

    def get_active_provider(self) -> LLMProvider:
        return self.provider

    def switch_provider(self, provider: LLMProvider | str) -> None:
        self.provider = LLMProvider(provider)

    def embed_texts(
        self, texts: list[str], provider_override: LLMProvider | str | None = None
    ) -> list[list[float]]:
        if self.slow_seconds > 0:
            time.sleep(self.slow_seconds)
        self.embed_calls.append(list(texts))
        if (
            self.fail_on_batch is not None
            and len(self.embed_calls) >= self.fail_on_batch
        ):
            raise ValueError("Simulated batch gateway failure")
        vectors = []
        for text in texts:
            vec = np.zeros(8, dtype=float)
            low = text.casefold()
            idx = (
                0
                if "payment" in low or "دفع" in text
                else 1
                if "delivery" in low
                else 2
            )
            vec[idx] = 1.0
            vec[3:] = (
                np.frombuffer(
                    hashlib.sha256(text.encode()).digest()[:5], dtype=np.uint8
                )
                / 25500.0
            )
            vectors.append(vec.tolist())
        return vectors

    def complete_structured(self, **kwargs):
        raise ValueError("Reranking disabled in test")


def _sample_records(n: int = 12) -> list[dict]:
    templates = [
        "My payment failed and I need help with this transaction",
        "My delivery has not arrived and I need help finding it",
    ]
    return [
        {"ticket_id": f"t-{i}-{j}", "raw_text": f"{tmpl} #{i}"}
        for i in range(n)
        for j, tmpl in enumerate(templates)
    ]


def _make_settings(tmp_path: Path, **overrides) -> AppSettings:
    cfg = {
        "release": SYNTHETIC_RELEASE_POLICY,
        "classification": {
            "enable_llm_reranker": False,
            "lexical_weight": 0.1,
            "metadata_weight": 0.0,
            "open_set_distance_quantile": 1.0,
            "candidate_models": [
                "naive_majority",
                "logistic_regression",
                "frozen_prototype",
            ],
        },
        "embedding": {"batch_size": 4, "dtype": "float32"},
        "admission": {
            "max_request_bytes": 512_000,
            "max_concurrent_jobs_per_tenant": 2,
        },
    }
    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return AppSettings(
        gateway_base="https://gateway.example/v1",
        gateway_key="test-key",
        embedding_dimensions=8,
        config_path=str(cfg_path),
        data_dir=str(tmp_path),
        _env_file=None,
        **overrides,
    )


def test_auth_tenant_isolation_and_workload_limits(tmp_path: Path) -> None:
    settings = _make_settings(
        tmp_path,
        auth_enabled=True,
        auth_tokens={
            "tok-reader-a": {
                "subject": "alice",
                "tenant_id": "tenant-a",
                "roles": ["reader"],
                "allowed_domains": ["generic"],
            },
            "tok-trainer-a": {
                "subject": "bob",
                "tenant_id": "tenant-a",
                "roles": ["trainer"],
                "allowed_domains": ["generic"],
            },
            "tok-admin-b": {
                "subject": "carol",
                "tenant_id": "tenant-b",
                "roles": ["admin"],
                "allowed_domains": ["*"],
            },
        },
    )
    gw = ControlledGateway()
    container = ServiceContainer(settings=settings, gateway=gw)
    client = TestClient(create_app(container))

    assert client.get("/api/v1/clusters").status_code == 401
    assert (
        client.get(
            "/api/v1/clusters", headers={"Authorization": "Bearer bad"}
        ).status_code
        == 401
    )

    reader_h = {"Authorization": "Bearer tok-reader-a"}
    assert (
        client.post(
            "/api/v1/fit",
            headers=reader_h,
            json={"records": _sample_records(6), "use_llm_naming": False},
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/v1/provider", headers=reader_h, json={"provider": "openai"}
        ).status_code
        == 403
    )

    assert (
        client.get(
            "/api/v1/clusters?domain_id=customer_claim", headers=reader_h
        ).status_code
        == 403
    )

    trainer_h = {"Authorization": "Bearer tok-trainer-a"}
    r_fit = client.post(
        "/api/v1/fit",
        headers=trainer_h,
        json={"records": _sample_records(10), "use_llm_naming": False},
    )
    assert r_fit.status_code == 202, r_fit.text
    assert FitWorker(container).run_once()
    assert (
        container.fit_job_repo.get_job(r_fit.json()["job_id"])["status"] == "completed"
    )

    admin_b_h = {"Authorization": "Bearer tok-admin-b"}
    assert (
        client.get("/api/v1/clusters", headers=admin_b_h).json()["cluster_count"] == 0
    )

    calls_before = len(gw.embed_calls)
    assert (
        client.post(
            "/api/v1/classify",
            headers=reader_h,
            json={"ticket_id": "big", "text": "x" * 33_000},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/classify",
            headers=reader_h,
            json={
                "ticket_id": "meta-big",
                "text": "payment failed",
                "metadata": {f"k{i}": i for i in range(70)},
            },
        ).status_code
        == 422
    )
    assert len(gw.embed_calls) == calls_before

    assert (
        client.post(
            "/api/v1/classify",
            headers={**reader_h, "Content-Type": "application/json"},
            content=b"{" + b" " * 600_000 + b"}",
        ).status_code
        == 413
    )


def test_async_fit_idempotency_and_nonblocking_classify(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    gw = ControlledGateway()
    container = ServiceContainer(settings=settings, gateway=gw)
    client = TestClient(create_app(container))

    assert (
        client.post(
            "/api/v1/fit",
            json={"records": _sample_records(10), "use_llm_naming": False},
        ).status_code
        == 200
    )

    gw.slow_seconds = 0.05
    r1 = client.post(
        "/api/v1/fit",
        json={
            "records": _sample_records(10),
            "use_llm_naming": False,
            "async_job": True,
            "idempotency_key": "job-key-1",
        },
    )
    assert r1.status_code == 202
    job_id = r1.json()["job_id"]

    r2 = client.post(
        "/api/v1/fit",
        json={
            "records": _sample_records(10),
            "use_llm_naming": False,
            "async_job": True,
            "idempotency_key": "job-key-1",
        },
    )
    assert r2.status_code == 202
    assert r2.json()["job_id"] == job_id

    worker_thread = Thread(target=FitWorker(container).run_once)
    worker_thread.start()

    clf = client.post(
        "/api/v1/classify",
        json={
            "ticket_id": "live-1",
            "text": "My payment failed and I need help with this transaction #0",
        },
    )
    assert clf.status_code == 200
    assert clf.json()["predicted_cluster_id"] >= 0

    for _ in range(60):
        status_resp = client.get(f"/api/v1/jobs/{job_id}").json()
        if status_resp["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)
    worker_thread.join(timeout=10)
    assert not worker_thread.is_alive()
    assert status_resp["status"] == "completed", status_resp.get("error")

    r3 = client.post(
        "/api/v1/fit",
        json={
            "records": _sample_records(10),
            "use_llm_naming": False,
            "async_job": True,
            "idempotency_key": "job-key-cancel",
        },
    )
    cancel_resp = client.post(f"/api/v1/jobs/{r3.json()['job_id']}/cancel")
    assert cancel_resp.status_code == 200


def test_true_batching_single_batch_parity_and_gateway_retries(
    tmp_path: Path,
) -> None:
    settings = _make_settings(tmp_path)
    gw = ControlledGateway()
    container = ServiceContainer(settings=settings, gateway=gw)
    client = TestClient(create_app(container))
    assert (
        client.post(
            "/api/v1/fit",
            json={"records": _sample_records(10), "use_llm_naming": False},
        ).status_code
        == 200
    )

    gw.embed_calls.clear()
    batch_payload = {
        "tickets": [
            {"ticket_id": "b-1", "text": "Brand new uncached payment issue alpha"},
            {"ticket_id": "b-2", "text": "Brand new uncached delivery issue beta"},
            {"ticket_id": "b-3", "text": "Brand new uncached payment issue alpha"},
        ]
    }
    r_batch = client.post("/api/v1/classify/batch", json=batch_payload)
    assert r_batch.status_code == 200

    assert len(gw.embed_calls) == 1
    assert len(gw.embed_calls[0]) == 2

    r_single = client.post(
        "/api/v1/classify",
        json={"ticket_id": "b-1", "text": "Brand new uncached payment issue alpha"},
    )
    assert r_single.status_code == 200
    assert (
        r_single.json()["predicted_cluster_id"]
        == r_batch.json()[0]["predicted_cluster_id"]
    )
    assert r_single.json()["confidence"] == r_batch.json()[0]["confidence"]

    mock_client = MagicMock()
    mock_client.embed.side_effect = [
        GatewaySDKError("429 rate limited"),
        MagicMock(vectors=[[0.1, 0.2]]),
    ]
    adapter = LLMGatewaySDKAdapter(settings=settings, client=mock_client, max_retries=2)
    vecs = adapter.embed_texts(["hello"])
    assert vecs == [[0.1, 0.2]]
    assert mock_client.embed.call_count == 2


def test_incremental_batch_persistence_and_bounded_discovery(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)

    gw = ControlledGateway(fail_on_batch=2)
    container = ServiceContainer(settings=settings, gateway=gw)
    svc = container.pipeline_service
    texts = [f"Unique payment ticket number {i}" for i in range(6)]
    with pytest.raises(ValueError, match="Simulated batch gateway failure"):
        svc.embed_with_dedup_and_cache(texts, batch_size=4)

    gw.fail_on_batch = None
    gw.embed_calls.clear()
    vecs, stats = svc.embed_with_dedup_and_cache(texts, batch_size=4)
    assert vecs.shape == (6, 8)
    assert vecs.dtype == np.float32
    assert stats["cache_hits"] == 4
    assert stats["gateway_embedded_texts"] == 2

    rng = np.random.default_rng(11)
    data = np.vstack([center + rng.normal(0, 0.02, (40, 6)) for center in np.eye(3, 6)])
    hints = ["en"] * 100 + ["ar"] * 20
    engine = ClusteringEngine(
        strategy="aligned_kmeans",
        reducer="none",
        max_discovery_samples=60,
        k_min=2,
        k_max=4,
    )
    labels, _ = engine.fit_transform(data, strata_hints=hints)
    assert len(labels) == 120
    assert engine.diagnostics_["discovery_mode"] == "sampled"
    assert engine.diagnostics_["discovery_sample_size"] == 60


def test_multiprocess_promotion_rollback_trust_and_fingerprints(
    tmp_path: Path,
) -> None:
    settings = _make_settings(tmp_path, artifact_hmac_secret="super-secret-hmac")
    gw = ControlledGateway()
    c1 = ServiceContainer(settings=settings, gateway=gw)
    c2 = ServiceContainer(settings=settings, gateway=gw)

    s1 = c1.pipeline_service
    s1.fit_dataframe(pd.DataFrame(_sample_records(10)), use_llm_naming=False)
    v1 = s1.taxonomy_version

    s1.fit_dataframe(pd.DataFrame(_sample_records(10)), use_llm_naming=False)
    v2 = s1.taxonomy_version
    assert v1 != v2

    s2 = c2.pipeline_service
    res = s2.classify_single_ticket("p-1", "My payment failed and I need help")
    assert res.taxonomy_version == v2

    s1.rollback_model()
    res_rb = s2.classify_single_ticket("p-2", "My payment failed and I need help")
    assert res_rb.taxonomy_version == v1

    with pytest.raises(ValueError, match="CAS promotion conflict"):
        s1.promote_model(target_version=v2, expected_previous_version="wrong-version")

    reg = ModelRegistryRepository(
        Path(settings.data_dir) / "artifacts/model_registry.joblib",
        hmac_secret="super-secret-hmac",
    )
    manifest_path = reg.versions_dir / f"{v1}.joblib.manifest.json"
    manifest_path.write_text(
        manifest_path.read_text().replace(v1, "tampered"), encoding="utf-8"
    )
    with pytest.raises(
        ValueError, match="integrity check failed|signature verification failed"
    ):
        reg.load_bundle(version=v1)

    id_before = s1.representation_identity()
    s1.profile.routes["general"] = "Renamed Support Queue"
    id_after = s1.representation_identity()
    assert id_before == id_after


def test_review_workflow_readiness_and_metrics(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    gw = ControlledGateway()
    container = ServiceContainer(settings=settings, gateway=gw)
    client = TestClient(create_app(container))

    assert client.get("/health").status_code == 200
    assert client.get("/health/ready?require_model=true").status_code == 503

    client.post(
        "/api/v1/fit",
        json={"records": _sample_records(10), "use_llm_naming": False},
    )
    assert client.get("/health/ready?require_model=true").status_code == 200

    for i in range(3):
        client.post("/api/v1/classify", json={"ticket_id": f"empty-{i}", "text": ""})

    client.post("/api/v1/classify", json={"ticket_id": "empty-0", "text": ""})

    page1 = client.get("/api/v1/review-items?limit=2").json()
    assert len(page1["items"]) == 2
    assert page1["next_cursor"] is not None
    page2 = client.get(
        f"/api/v1/review-items?limit=2&cursor={page1['next_cursor']}"
    ).json()
    assert len(page2["items"]) == 1
    assert page2["next_cursor"] is None

    adj = client.post(
        "/api/v1/review-items/empty-0/adjudicate",
        json={
            "status": "approved",
            "adjudicated_label": "payment_issue",
            "approved_for_training": True,
            "reviewer": "qa-lead",
        },
    )
    assert adj.status_code == 200
    assert adj.json()["approved_for_training"] is True
    assert adj.json()["occurrence_count"] == 2

    metrics = client.get("/metrics").json()
    assert "peak_rss_mb" in metrics
    assert "job_backlog" in metrics
    assert "review_backlog" in metrics
    assert "counters" in metrics
