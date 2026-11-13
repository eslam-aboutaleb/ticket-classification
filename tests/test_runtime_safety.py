"""Artifact isolation, promotion fencing, job recovery and release authorization."""

from __future__ import annotations

import asyncio
import multiprocessing
import shutil
import sqlite3
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_production_readiness import ControlledGateway, _make_settings, _sample_records

from ticket_classification.config.pipeline import DomainProfile, ReleasePolicy
from ticket_classification.controllers.api import ServiceContainer, create_app
from ticket_classification.controllers.request_limits import RequestSizeLimitMiddleware
from ticket_classification.models.schemas import (
    FitRequest,
    PromoteModelRequest,
    RollbackModelRequest,
)
from ticket_classification.repositories.fit_job_repo import (
    FitJobRepository,
    LeaseLostError,
)
from ticket_classification.repositories.model_registry_repo import (
    ModelRegistryRepository,
)
from ticket_classification.services.fit_worker import FitWorker
from ticket_classification.services.release_policy import assess_release


def _bundle(version, tenant="default", domain="generic", gate=None):
    return {
        "taxonomy_version": version,
        "tenant_id": tenant,
        "domain_id": domain,
        "release_gate": gate,
    }


@pytest.mark.parametrize(
    "version",
    [
        "../victim",
        "/tmp/victim",
        "../../B/versions/v1",
        r"..\victim",
        ".",
        "v1.research",
        "v\x00",
        "",
    ],
)
def test_versions_reject_path_input_at_schema_and_repository(tmp_path, version):
    repo = ModelRegistryRepository(tmp_path / "registry.joblib")
    with pytest.raises(ValidationError):
        PromoteModelRequest(target_version=version)
    with pytest.raises(ValidationError):
        RollbackModelRequest(target_version=version)
    for operation in (repo.promote_version, repo.load_bundle):
        with pytest.raises(ValueError, match="Invalid model version"):
            operation(version)
    with pytest.raises(ValueError, match="Invalid model version"):
        repo.save_bundle(_bundle(version or "../invalid"))


def test_symlinks_and_foreign_provenance_rejected_before_deserialization(
    tmp_path, monkeypatch
):
    a = ModelRegistryRepository(
        tmp_path / "A/model.joblib",
        hmac_secret="shared",
        tenant_id="A",
        domain_id="generic",
    )
    b = ModelRegistryRepository(
        tmp_path / "B/model.joblib",
        hmac_secret="shared",
        tenant_id="B",
        domain_id="generic",
    )
    b.save_bundle(_bundle("v1", "B"))
    a.versions_dir.mkdir()
    (a.versions_dir / "linked.joblib").symlink_to(b.version_path("v1"))
    with pytest.raises(ValueError, match="escapes"):
        a.load_bundle("linked")
    shutil.copyfile(b.version_path("v1"), a.version_path("v1"))
    shutil.copyfile(
        b.version_path("v1").with_suffix(".joblib.manifest.json"),
        a.version_path("v1").with_suffix(".joblib.manifest.json"),
    )
    loader = Mock(side_effect=AssertionError("must verify before loading"))
    monkeypatch.setattr(
        "ticket_classification.repositories.model_registry_repo.joblib.load", loader
    )
    with pytest.raises(ValueError, match="different tenant"):
        a.load_bundle("v1")
    with pytest.raises(ValueError, match="different tenant"):
        a.promote_version("v1")
    loader.assert_not_called()
    other_domain = ModelRegistryRepository(
        b.registry_path, hmac_secret="shared", tenant_id="B", domain_id="customer_claim"
    )
    with pytest.raises(ValueError, match="different domain"):
        other_domain.promote_version("v1")


def _competing_promotion(path, version, start, results):
    repo = ModelRegistryRepository(path)
    start.wait(20)
    try:
        repo.promote_version(version, expected_previous_version="v0")
        results.put("promoted")
    except ValueError as exc:
        results.put(str(exc))


def test_real_processes_fence_competing_promotions_and_preserve_active(tmp_path):
    path = tmp_path / "model.joblib"
    repo = ModelRegistryRepository(path)
    repo.save_bundle(_bundle("v0"))
    for version in ("v1", "v2"):
        repo.save_bundle(_bundle(version), promote=False)
    ctx = multiprocessing.get_context("spawn")
    start, results = ctx.Event(), ctx.Queue()
    processes = [
        ctx.Process(target=_competing_promotion, args=(path, version, start, results))
        for version in ("v1", "v2")
    ]
    try:
        for process in processes:
            process.start()
        start.set()
        outcomes = [results.get(timeout=30) for _ in processes]
        assert outcomes.count("promoted") == 1
        assert sum("CAS promotion conflict" in result for result in outcomes) == 1
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
    assert all(process.exitcode == 0 for process in processes)
    active = repo.get_release_record()["active_version"]
    with pytest.raises(ValueError, match="CAS promotion conflict"):
        repo.save_bundle(_bundle("stale"), expected_previous_version="v0")
    assert repo.load_bundle()["taxonomy_version"] == active
    empty = ModelRegistryRepository(tmp_path / "empty/model.joblib")
    empty.save_bundle(_bundle("candidate"), promote=False)
    with pytest.raises(ValueError, match="CAS promotion conflict"):
        empty.promote_version("candidate", expected_previous_version="v0")


def test_job_routes_enforce_domains_before_read_and_cancel(tmp_path):
    settings = _make_settings(
        tmp_path,
        auth_enabled=True,
        auth_tokens={
            "restricted": {
                "tenant_id": "A",
                "roles": ["trainer"],
                "allowed_domains": ["generic"],
            }
        },
    )
    container = ServiceContainer(settings, ControlledGateway())
    job, _ = container.fit_job_repo.enqueue("A", "customer_claim", {"records": []})
    with TestClient(create_app(container)) as client:
        headers = {"Authorization": "Bearer restricted"}
        path = "/api/v1/jobs/" + job["job_id"]
        assert client.get(path, headers=headers).status_code == 403
        assert client.post(path + "/cancel", headers=headers).status_code == 403
        assert container.fit_job_repo.get_job(job["job_id"])["status"] == "queued"


@pytest.mark.parametrize("headers", [[], [(b"content-length", b"1")]])
def test_chunked_or_underdeclared_body_stops_reading_at_limit(headers):
    calls, consumed, sent = [], [], []

    async def downstream(scope, receive, send):
        calls.append(True)

    async def receive():
        consumed.append(True)
        return {
            "type": "http.request",
            "body": b"x" * 256,
            "more_body": len(consumed) < 20,
        }

    async def send(message):
        sent.append(message)

    middleware = RequestSizeLimitMiddleware(downstream, max_bytes=lambda: 1024)
    asyncio.run(middleware({"type": "http", "headers": headers}, receive, send))
    assert len(consumed) == 5
    assert not calls
    assert sent[0]["status"] == 413


def test_within_limit_stream_replayed_exactly_and_disconnect_handled():
    chunks = iter(
        [
            {"type": "http.request", "body": b"abc", "more_body": True},
            {"type": "http.request", "body": b"def", "more_body": False},
            {"type": "http.disconnect"},
        ]
    )
    seen = []

    async def receive():
        return next(chunks)

    async def send(message):
        pass

    async def downstream(scope, receive, send):
        seen.append(await receive())
        seen.append(await receive())

    asyncio.run(
        RequestSizeLimitMiddleware(downstream, lambda: 6)(
            {"type": "http", "headers": []}, receive, send
        )
    )
    assert seen == [
        {"type": "http.request", "body": b"abcdef", "more_body": False},
        {"type": "http.disconnect"},
    ]


def _expire(repo, job_id):
    with sqlite3.connect(repo.db_path) as conn:
        conn.execute(
            "UPDATE fit_jobs SET lease_expires_at = 0 WHERE job_id = ?", (job_id,)
        )


def test_lease_reclaim_fences_updates_checkpoints_and_publication(tmp_path):
    repo = FitJobRepository(tmp_path / "jobs.sqlite")
    job, _ = repo.enqueue("A", "generic", {})
    assert repo.claim_next("old")["job_id"] == job["job_id"]
    assert repo.claim_next("other") is None
    assert repo.renew_lease(job["job_id"], "old")
    _expire(repo, job["job_id"])
    assert repo.claim_next("new")["job_id"] == job["job_id"]
    assert not repo.renew_lease(job["job_id"], "old")
    with pytest.raises(LeaseLostError):
        repo.update_stage(job["job_id"], "completed", "done", owner_id="old")
    with pytest.raises(LeaseLostError):
        repo.save_checkpoint(
            "A",
            "generic",
            "candidate",
            "hash",
            {},
            job_id=job["job_id"],
            owner_id="old",
        )
    with pytest.raises(LeaseLostError), repo.publication_guard(job["job_id"], "old"):
        pytest.fail("stale worker entered publication")
    assert repo.owns_lease(job["job_id"], "new")
    repo.cancel_job(job["job_id"], "A")
    assert not repo.owns_lease(job["job_id"], "new")


def test_queued_job_survives_container_restart_and_worker_completes_it(tmp_path):
    settings = _make_settings(tmp_path)
    first = ServiceContainer(settings, ControlledGateway())
    payload = FitRequest(
        records=_sample_records(10),
        use_llm_naming=False,
        async_job=True,
        idempotency_key="restart",
    )
    job = first.submit_fit_job("default", "generic", payload)
    first.close()
    second = ServiceContainer(settings, ControlledGateway())
    try:
        assert (
            second.submit_fit_job("default", "generic", payload)["job_id"]
            == job["job_id"]
        )
        worker = FitWorker(second)
        assert worker.run_once()
        done = second.fit_job_repo.get_job(job["job_id"])
        assert done["status"] == "completed", done.get("error")
        assert done["result"]["publication"]["status"] == "active"
        assert not worker.run_once()
    finally:
        second.close()


class _SimulatedProcessDeath(BaseException):
    pass


def test_worker_recovers_candidate_checkpoint_without_repeating_training(
    tmp_path, monkeypatch
):
    settings = _make_settings(tmp_path)
    first = ServiceContainer(settings, ControlledGateway())
    payload = FitRequest(
        records=_sample_records(10), use_llm_naming=False, async_job=True
    )
    job = first.submit_fit_job("default", "generic", payload)
    monkeypatch.setattr(
        first.pipeline_service,
        "_activate_candidate",
        Mock(side_effect=_SimulatedProcessDeath),
    )
    with pytest.raises(_SimulatedProcessDeath):
        FitWorker(first).run_once()
    assert first.fit_job_repo.get_job(job["job_id"])["status"] == "running"
    _expire(first.fit_job_repo, job["job_id"])
    first.close()
    gateway = ControlledGateway()
    second = ServiceContainer(settings, gateway)
    try:
        assert FitWorker(second).run_once()
        result = second.fit_job_repo.get_job(job["job_id"])
        assert result["status"] == "completed", result.get("error")
        assert result["result"]["publication"]["status"] == "active"
        assert gateway.embed_calls == []
        assert (
            len(list(second.model_registry_repo.versions_dir.glob("*.research.joblib")))
            == 1
        )
    finally:
        second.close()


def test_failed_release_candidate_keeps_active_model_and_cannot_be_forced(tmp_path):
    settings = _make_settings(tmp_path)
    container = ServiceContainer(settings, ControlledGateway())
    try:
        frame = pd.DataFrame(_sample_records(10))
        active = container.pipeline_service.fit_dataframe(frame, use_llm_naming=False)[
            "taxonomy_version"
        ]
        config_path = Path(settings.config_path)
        config = yaml.safe_load(config_path.read_text())
        config.pop("release")
        config_path.write_text(yaml.safe_dump(config))
        candidate = container.pipeline_service.fit_dataframe(
            frame, use_llm_naming=False
        )
        assert candidate["publication"]["status"] == "candidate"
        assert (
            "insufficient_language_samples:ar" in candidate["release_gate"]["blockers"]
        )
        assert container.pipeline_service._capture_snapshot().taxonomy_version == active
        with pytest.raises(ValueError, match="failed release checks"):
            container.pipeline_service.promote_model(
                candidate["taxonomy_version"], approved_by="admin"
            )
        assert (
            container.model_registry_repo.get_release_record()["active_version"]
            == active
        )
    finally:
        container.close()


def test_human_approval_requires_admin_and_records_authenticated_reviewer(tmp_path):
    settings = _make_settings(
        tmp_path,
        auth_enabled=True,
        auth_tokens={
            "trainer": {"roles": ["trainer"]},
            "admin": {"subject": "alice", "roles": ["admin"]},
        },
    )
    path = Path(settings.config_path)
    config = yaml.safe_load(path.read_text())
    config["release"]["require_taxonomy_approval"] = True
    path.write_text(yaml.safe_dump(config))
    container = ServiceContainer(settings, ControlledGateway())
    metrics = container.pipeline_service.fit_dataframe(
        pd.DataFrame(_sample_records(10)), use_llm_naming=False
    )
    assert metrics["publication"]["status"] == "candidate"
    assert container.pipeline_service._capture_snapshot() is None
    with TestClient(create_app(container)) as client:
        payload = {"target_version": metrics["taxonomy_version"]}
        trainer = {"Authorization": "Bearer trainer"}
        assert (
            client.post(
                "/api/v1/models/promote", json=payload, headers=trainer
            ).status_code
            == 409
        )
        payload["approve_taxonomy"] = True
        assert (
            client.post(
                "/api/v1/models/promote", json=payload, headers=trainer
            ).status_code
            == 403
        )
        response = client.post(
            "/api/v1/models/promote",
            json=payload,
            headers={"Authorization": "Bearer admin"},
        )
        assert response.status_code == 200, response.text
        assert (
            response.json()["approvals"][metrics["taxonomy_version"]]["reviewer"]
            == "alice"
        )
        assert (
            container.pipeline_service._capture_snapshot().taxonomy_version
            == metrics["taxonomy_version"]
        )


def test_release_policy_checks_language_quality_calibration_and_unknowns():
    metrics = {
        "evaluated": True,
        "classification": {
            "calibration": "held_out_known",
            "slice_error_analysis": {"open_set_oos_recall": 0.9},
        },
        "release_gate": {
            "supported_language_coverage": {
                "en": {
                    "count": 5,
                    "cluster_agreement_macro_f1": 0.9,
                    "accepted_coverage": 0.8,
                }
            }
        },
    }
    profile = DomainProfile()
    profile.language.supported = ["en"]
    assert assess_release(metrics, profile, ReleasePolicy(min_oos_recall=0.8))["passed"]
    metrics["classification"]["calibration"] = "low_evidence_held_out"
    metrics["classification"]["slice_error_analysis"]["open_set_oos_recall"] = 0.1
    metrics["release_gate"]["supported_language_coverage"]["en"][
        "accepted_coverage"
    ] = float("nan")
    gate = assess_release(metrics, profile, ReleasePolicy(min_oos_recall=0.8))
    assert not gate["passed"]
    assert "held_out_calibration_required" in gate["blockers"]
    assert "unknown_intent_recall_below_threshold" in gate["blockers"]
    assert "language_quality:en:accepted_coverage" in gate["blockers"]


def _claim_until_killed(path, ready):
    repo = FitJobRepository(path)
    assert repo.claim_next("killed-process") is not None
    ready.set()

    multiprocessing.Event().wait(60)


def test_killed_worker_is_replaced_and_job_is_not_lost(tmp_path):
    settings = _make_settings(tmp_path)
    container = ServiceContainer(settings, ControlledGateway())
    job = container.submit_fit_job(
        "default",
        "generic",
        FitRequest(records=_sample_records(10), use_llm_naming=False),
    )
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    process = ctx.Process(
        target=_claim_until_killed, args=(container.fit_job_repo.db_path, ready)
    )
    try:
        process.start()
        assert ready.wait(20)
        process.kill()
        process.join(timeout=10)
        assert process.exitcode != 0
        _expire(container.fit_job_repo, job["job_id"])
        assert FitWorker(container).run_once()
        result = container.fit_job_repo.get_job(job["job_id"])
        assert result["status"] == "completed", result.get("error")
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        container.close()


def test_worker_failure_is_durable_and_live_attempt_renews_lease(tmp_path, monkeypatch):
    from threading import Event, Thread

    container = ServiceContainer(_make_settings(tmp_path), ControlledGateway())
    payload = FitRequest(records=[{"raw_text": "ticket"}])
    failed_job = container.submit_fit_job("default", "generic", payload)
    monkeypatch.setattr(
        container.pipeline_service,
        "fit_dataframe",
        Mock(side_effect=ValueError("bad input")),
    )
    assert FitWorker(container).run_once()
    result = container.fit_job_repo.get_job(failed_job["job_id"])
    assert result["status"] == "failed"
    assert result["error"] == "bad input"
    renewed, finish = Event(), Event()
    original_renew = container.fit_job_repo.renew_lease

    def renew(*args):
        result = original_renew(*args)
        if result:
            renewed.set()
        return result

    def long_fit(*args, **kwargs):
        assert finish.wait(5)
        return {"publication": {"status": "candidate"}}

    monkeypatch.setattr(container.fit_job_repo, "renew_lease", renew)
    monkeypatch.setattr(container.pipeline_service, "fit_dataframe", long_fit)
    job = container.submit_fit_job("default", "generic", payload)
    thread = Thread(target=FitWorker(container, lease_seconds=3).run_once)
    thread.start()
    try:
        assert renewed.wait(5)
        assert container.fit_job_repo.get_job(job["job_id"])["status"] == "running"
    finally:
        finish.set()
        thread.join(timeout=5)
        container.close()
    assert not thread.is_alive()
    assert container.fit_job_repo.get_job(job["job_id"])["status"] == "completed"


def test_business_release_thresholds_require_measured_evidence():
    profile = DomainProfile()
    policy = ReleasePolicy(
        require_evaluation=False,
        require_calibration=False,
        min_language_samples=0,
        min_ground_truth_macro_f1=0.85,
        max_known_false_positive_rate=0.05,
    )
    assert not assess_release({}, profile, policy)["passed"]
    metrics = {
        "classification": {
            "selected_production_model": "logistic_regression",
            "model_comparison": {
                "logistic_regression": {"test_ground_truth_macro_f1": 0.9}
            },
            "slice_error_analysis": {
                "open_set_evaluation": {"known_false_positive_rate": 0.03}
            },
        }
    }
    assert assess_release(metrics, profile, policy)["passed"]
