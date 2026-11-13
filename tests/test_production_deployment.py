"""Production startup, deployment isolation and operational behavior."""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_production_readiness import ControlledGateway

from ticket_classification.config.settings import AppSettings
from ticket_classification.controllers.api import ServiceContainer, create_app
from ticket_classification.controllers.healthcheck import main as healthcheck
from ticket_classification.repositories.fit_job_repo import FitJobRepository
from ticket_classification.services.fit_worker import FitWorker

TOKEN = "test-deployment-token-" + "a" * 32


def _options(tmp_path):
    config = tmp_path / "pipeline.yaml"
    config.write_text("{}\n")
    return {
        "app_env": "production",
        "auth_enabled": True,
        "auth_tokens": {
            TOKEN: {
                "subject": "operator",
                "tenant_id": "tenant-a",
                "roles": ["admin"],
                "allowed_domains": ["*"],
            }
        },
        "tenant_id": "tenant-a",
        "artifact_hmac_secret": "s" * 48,
        "embedding_revision": "embedding-release-1",
        "config_path": str(config),
        "data_dir": str(tmp_path / "data"),
        "gateway_base": "https://gateway.example/v1",
        "gateway_key": "sk-test-gateway-key",
        "allowed_hosts": ["testserver", "localhost", "127.0.0.1"],
        "_env_file": None,
    }


@pytest.mark.parametrize(
    "override",
    [
        {"auth_enabled": False},
        {"auth_tokens": {}},
        {"artifact_hmac_secret": "short"},
        {"embedding_revision": ""},
        {"allowed_hosts": ["*"]},
        {"config_path": "/missing/pipeline.yaml"},
        {"data_dir": "relative/data"},
        {"gateway_base": "http://gateway.example/v1"},
        {"gateway_verify_ssl": False},
        {"gateway_key": None},
        {"gateway_max_retries": 2},
        {"gateway_timeout": 0},
        {"gateway_timeout": 121},
    ],
)
def test_production_fails_closed_for_unsafe_configuration(tmp_path, override):
    with pytest.raises(ValidationError) as error:
        AppSettings(**{**_options(tmp_path), **override})
    assert TOKEN not in str(error.value)
    assert "sk-test-gateway-key" not in str(error.value)


@pytest.mark.parametrize(
    "principal",
    [
        {"tenant_id": "../escape"},
        {"roles": ["superuser"]},
        {"roles": []},
        {"allowed_domains": ["customer_*"]},
        {"allowed_domains": []},
    ],
)
def test_principal_configuration_rejects_invalid_scope(tmp_path, principal):
    options = _options(tmp_path)
    base = options["auth_tokens"][TOKEN]
    options["auth_tokens"] = {TOKEN: {**base, **principal}}
    with pytest.raises(ValidationError):
        AppSettings(**options)


def test_mounted_secrets_are_loaded_and_redacted(tmp_path):
    options = _options(tmp_path)
    token_file, hmac_file, gateway_file = [
        tmp_path / name for name in ("tokens.json", "hmac", "gateway")
    ]
    token_file.write_text(json.dumps(options.pop("auth_tokens")))
    hmac_file.write_text(options.pop("artifact_hmac_secret"))
    gateway_file.write_text("sk-mounted-test-key\n")
    settings = AppSettings(
        **{
            **options,
            "auth_tokens_file": token_file,
            "artifact_hmac_secret_file": hmac_file,
            "gateway_key_file": gateway_file,
        }
    )
    assert TOKEN in settings.auth_tokens
    assert settings.gateway_key.get_secret_value() == "sk-mounted-test-key"
    assert settings.artifact_hmac_secret == "s" * 48
    for secret in (TOKEN, "sk-mounted-test-key", "s" * 48):
        assert secret not in repr(settings)
    token_file.write_text("[]")
    with pytest.raises(ValidationError):
        AppSettings(**{**options, "auth_tokens_file": token_file})


def test_production_hosts_readiness_and_docs(tmp_path):
    settings = AppSettings(**_options(tmp_path))
    container = ServiceContainer(settings, ControlledGateway())
    with TestClient(create_app(container)) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.headers["x-content-type-options"] == "nosniff"
        assert len(response.headers["x-request-id"]) == 32
        assert (
            client.get("/health", headers={"Host": "attacker.example"}).status_code
            == 400
        )
        assert client.get("/health/ready?require_model=false").status_code == 503
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404
        assert (
            client.post(
                "/api/v1/classify", json={"text": "a private ticket"}
            ).status_code
            == 401
        )


def test_metrics_require_admin_and_never_expose_another_tenant(tmp_path):
    options = _options(tmp_path)
    options["auth_tokens"].update(
        {
            "reader-" + "b" * 32: {
                "subject": "reader",
                "roles": ["reader"],
                "tenant_id": "tenant-a",
            },
            "restricted-" + "c" * 32: {
                "subject": "restricted",
                "roles": ["admin"],
                "tenant_id": "tenant-a",
                "allowed_domains": ["generic"],
            },
        }
    )
    container = ServiceContainer(AppSettings(**options), ControlledGateway())
    other = container.get_service(tenant_id="tenant-b")
    other.telemetry.inc("private_tenant_b_counter", 42)
    other.telemetry.set_active_version("tenant-b", "generic", "private-version")
    with TestClient(create_app(container)) as client:
        assert (
            client.get(
                "/metrics", headers={"Authorization": "Bearer reader-" + "b" * 32}
            ).status_code
            == 403
        )
        assert (
            client.get(
                "/metrics", headers={"Authorization": "Bearer restricted-" + "c" * 32}
            ).status_code
            == 403
        )
        metrics = client.get("/metrics", headers={"Authorization": "Bearer " + TOKEN})
        assert metrics.status_code == 200
        assert "private_tenant_b_counter" not in metrics.text
        assert "private-version" not in metrics.text


def test_poison_job_stops_after_recovery_limit(tmp_path):
    repo = FitJobRepository(tmp_path / "jobs.sqlite")
    job, _ = repo.enqueue("default", "generic", {})
    for attempt in range(1, 4):
        claimed = repo.claim_next(f"worker-{attempt}", max_attempts=3)
        assert claimed["attempts"] == attempt
        with sqlite3.connect(repo.db_path) as conn:
            conn.execute(
                "UPDATE fit_jobs SET lease_expires_at = 0 WHERE job_id = ?",
                (job["job_id"],),
            )
    assert repo.claim_next("replacement", max_attempts=3) is None
    failed = repo.get_job(job["job_id"])
    assert failed["status"] == "failed"
    assert failed["error"] == "Maximum recovery attempts exceeded"


def test_worker_probe_tracks_consumer_heartbeat(tmp_path, monkeypatch):
    settings = AppSettings(**{**_options(tmp_path), "app_env": "test"})
    container = ServiceContainer(settings, ControlledGateway())
    health_file = tmp_path / "worker-heartbeat"
    monkeypatch.setenv("WORKER_HEALTH_FILE", str(health_file))
    try:
        assert healthcheck(["worker"]) == 1
        assert FitWorker(container, health_file=health_file).run_once() is False
        assert healthcheck(["worker"]) == 0
        os.utime(health_file, (time.time() - 100, time.time() - 100))
        assert healthcheck(["worker"]) == 1
    finally:
        container.close()


def test_native_server_starts_securely_and_logs_no_request_secrets(tmp_path):
    options = _options(tmp_path)
    env = os.environ.copy()
    for key, value in options.items():
        if key == "_env_file":
            continue
        env[key.upper()] = (
            json.dumps(value) if isinstance(value, (dict, list, bool)) else str(value)
        )
    env["LLM_GATEWAY_BASE_URL"] = options["gateway_base"]
    env["LLM_GATEWAY_API_KEY"] = options["gateway_key"]
    for key in ("AUTH_TOKENS_FILE", "ARTIFACT_HMAC_SECRET_FILE", "GATEWAY_KEY_FILE"):
        env.pop(key, None)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "ticket_classification.controllers.server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=1) as client:
            deadline = time.monotonic() + 15
            while True:
                try:
                    result = client.get("/health?private_query=never-log-this")
                    if result.status_code == 200:
                        break
                except httpx.ConnectError:
                    pass
                if process.poll() is not None or time.monotonic() > deadline:
                    pytest.fail("Production API failed to start")
                time.sleep(0.05)
            assert client.get("/health/ready").status_code == 503
            assert client.get("/api/v1/clusters").status_code == 401
            assert (
                client.get(
                    "/api/v1/clusters", headers={"Authorization": "Bearer " + TOKEN}
                ).status_code
                == 200
            )
    finally:
        process.terminate()
        try:
            output, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
    assert TOKEN not in output
    assert "never-log-this" not in output
    events = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    assert any(
        event.get("route") == "/health" and event.get("status") == 200
        for event in events
    )
