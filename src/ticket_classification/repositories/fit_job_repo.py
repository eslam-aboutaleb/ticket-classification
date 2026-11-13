"""Durable SQLite repository for background training jobs, idempotency deduplication, and stage checkpoints."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


class LeaseLostError(RuntimeError):
    """The job was cancelled or this worker no longer owns its lease."""


class FitJobRepository:
    """Stores durable fit jobs and content-addressed stage checkpoints."""

    def __init__(
        self,
        db_path: str | Path = "data/fit_jobs.sqlite",
        busy_timeout_ms: int = 5000,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.busy_timeout_ms = busy_timeout_ms
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=self.busy_timeout_ms / 1000.0)
        try:
            conn.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fit_jobs (
                    job_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    domain_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    progress_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (tenant_id, domain_id, idempotency_key)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stage_checkpoints (
                    checkpoint_key TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    domain_id TEXT NOT NULL,
                    stage_name TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(fit_jobs)")}
            for name, kind in [
                ("lease_owner", "TEXT"),
                ("lease_expires_at", "REAL"),
                ("attempts", "INTEGER NOT NULL DEFAULT 0"),
            ]:
                if name not in columns:
                    conn.execute(f"ALTER TABLE fit_jobs ADD COLUMN {name} {kind}")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS fit_job_claim ON fit_jobs(status, lease_expires_at, created_at)"
            )
            conn.commit()

    @staticmethod
    def compute_idempotency_key(
        tenant_id: str,
        domain_id: str,
        payload: dict[str, Any],
        explicit_key: str | None = None,
    ) -> str:
        if explicit_key:
            return explicit_key
        raw = json.dumps(
            {"tenant_id": tenant_id, "domain_id": domain_id, **payload},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def enqueue(
        self,
        tenant_id: str,
        domain_id: str,
        payload: dict[str, Any],
        idempotency_key: str | None = None,
        max_concurrent_jobs: int = 2,
    ) -> tuple[dict[str, Any], bool]:
        """Enqueue a fit job or return an existing active/completed job with the same idempotency_key.

        Returns (job_dict, is_new_job).
        """
        idem = self.compute_idempotency_key(
            tenant_id, domain_id, payload, idempotency_key
        )
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                """
                SELECT * FROM fit_jobs
                WHERE tenant_id = ? AND domain_id = ? AND idempotency_key = ?
                """,
                (tenant_id, domain_id, idem),
            ).fetchone()
            if existing and existing["status"] in {"queued", "running", "completed"}:
                return self._format_row(dict(existing)), False

            active_count = conn.execute(
                """
                SELECT COUNT(*) FROM fit_jobs
                WHERE tenant_id = ? AND status IN ('queued', 'running')
                """,
                (tenant_id,),
            ).fetchone()[0]
            if int(active_count) >= max_concurrent_jobs:
                raise RuntimeError(
                    f"Concurrent fit job limit ({max_concurrent_jobs}) reached for tenant '{tenant_id}'"
                )

            if existing:
                job_id = str(existing["job_id"])
                conn.execute(
                    """
                    UPDATE fit_jobs
                    SET status = 'queued', stage = 'queued', payload_json = ?,
                        progress_json = '{}', result_json = NULL, error = NULL, attempts = 0, lease_owner = NULL, lease_expires_at = NULL, updated_at = ?
                    WHERE job_id = ?
                    """,
                    (json.dumps(payload, default=str), now, job_id),
                )
            else:
                job_id = uuid4().hex
                conn.execute(
                    """
                    INSERT INTO fit_jobs
                    (job_id, tenant_id, domain_id, idempotency_key, status, stage, payload_json, progress_json, result_json, error, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 'queued', 'queued', ?, '{}', NULL, NULL, ?, ?)
                    """,
                    (
                        job_id,
                        tenant_id,
                        domain_id,
                        idem,
                        json.dumps(payload, default=str),
                        now,
                        now,
                    ),
                )
            row = conn.execute(
                "SELECT * FROM fit_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            conn.commit()
        return self._format_row(dict(row)), True

    def get_job(
        self, job_id: str, tenant_id: str | None = None
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            if tenant_id is not None:
                row = conn.execute(
                    "SELECT * FROM fit_jobs WHERE job_id = ? AND tenant_id = ?",
                    (job_id, tenant_id),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM fit_jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
        return self._format_row(dict(row)) if row else None

    def update_stage(
        self,
        job_id: str,
        status: str,
        stage: str,
        progress: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        owner_id: str | None = None,
    ) -> dict[str, Any] | None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM fit_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if not row:
                return None
            if owner_id is not None and (
                row["lease_owner"] != owner_id
                or (row["lease_expires_at"] or 0) <= time.time()
                or row["status"] != "running"
            ):
                raise LeaseLostError("Fit job lease lost")
            # Preserve cancelled status unless explicitly resetting
            if row["status"] == "cancelled" and status not in {"cancelled", "queued"}:
                return self._format_row(dict(row))
            merged_progress = {
                **json.loads(row["progress_json"] or "{}"),
                **(progress or {}),
            }
            conn.execute(
                """
                UPDATE fit_jobs
                SET status = ?, stage = ?, progress_json = ?,
                    result_json = COALESCE(?, result_json),
                    error = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    status,
                    stage,
                    json.dumps(merged_progress, default=str),
                    json.dumps(result, default=str) if result is not None else None,
                    error,
                    now,
                    job_id,
                ),
            )
            updated = conn.execute(
                "SELECT * FROM fit_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            conn.commit()
        return self._format_row(dict(updated))

    def cancel_job(
        self, job_id: str, tenant_id: str | None = None
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            query = "SELECT * FROM fit_jobs WHERE job_id = ?"
            params = [job_id]
            if tenant_id is not None:
                query += " AND tenant_id = ?"
                params.append(tenant_id)
            row = conn.execute(query, params).fetchone()
            if row is None:
                return None
            if row["status"] not in {"completed", "failed", "cancelled"}:
                conn.execute(
                    """UPDATE fit_jobs SET status = 'cancelled', error = 'Cancelled by operator',
                       lease_owner = NULL, lease_expires_at = NULL, updated_at = ? WHERE job_id = ?""",
                    (datetime.now(UTC).isoformat(), job_id),
                )
                row = conn.execute(query, params).fetchone()
            return self._format_row(dict(row))

    def is_cancelled(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        return bool(job and job["status"] == "cancelled")

    def claim_next(
        self, owner_id: str, lease_seconds: float = 60, max_attempts: int = 3
    ) -> dict[str, Any] | None:
        """Atomically claim queued work or recover an expired/legacy running job."""
        now = time.time()
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """UPDATE fit_jobs SET status = 'failed', stage = 'failed',
                   error = 'Maximum recovery attempts exceeded', lease_owner = NULL, lease_expires_at = NULL,
                   updated_at = ? WHERE attempts >= ? AND
                   (status = 'queued' OR (status = 'running' AND COALESCE(lease_expires_at, 0) <= ?))""",
                (datetime.now(UTC).isoformat(), max_attempts, now),
            )
            row = conn.execute(
                """SELECT * FROM fit_jobs WHERE status = 'queued'
                   OR (status = 'running' AND COALESCE(lease_expires_at, 0) <= ?)
                   ORDER BY created_at, job_id LIMIT 1""",
                (now,),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                """UPDATE fit_jobs SET status = 'running', attempts = attempts + 1, lease_owner = ?, lease_expires_at = ?,
                   updated_at = ? WHERE job_id = ?""",
                (
                    owner_id,
                    now + lease_seconds,
                    datetime.now(UTC).isoformat(),
                    row["job_id"],
                ),
            )
            claimed = conn.execute(
                "SELECT * FROM fit_jobs WHERE job_id = ?", (row["job_id"],)
            ).fetchone()
            return self._format_row(dict(claimed))

    def renew_lease(
        self, job_id: str, owner_id: str, lease_seconds: float = 60
    ) -> bool:
        now = time.time()
        with self._connect() as conn:
            result = conn.execute(
                """UPDATE fit_jobs SET lease_expires_at = ?, updated_at = ?
                   WHERE job_id = ? AND lease_owner = ? AND status = 'running' AND lease_expires_at > ?""",
                (
                    now + lease_seconds,
                    datetime.now(UTC).isoformat(),
                    job_id,
                    owner_id,
                    now,
                ),
            )
            return result.rowcount == 1

    def owns_lease(self, job_id: str, owner_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM fit_jobs WHERE job_id = ? AND lease_owner = ? AND status = 'running' AND lease_expires_at > ?",
                (job_id, owner_id, time.time()),
            ).fetchone()
            return row is not None

    @contextmanager
    def publication_guard(self, job_id: str, owner_id: str):
        """Fence cancellation/reclaim while checking ownership and committing a model release."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT 1 FROM fit_jobs WHERE job_id = ? AND lease_owner = ? AND status = 'running' AND lease_expires_at > ?",
                (job_id, owner_id, time.time()),
            ).fetchone()
            if row is None:
                raise LeaseLostError("Fit job lease lost before publication")
            yield

    def list_jobs(
        self, tenant_id: str = "default", domain_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        clauses = ["tenant_id = ?"]
        params: list[Any] = [tenant_id]
        if domain_id:
            clauses.append("domain_id = ?")
            params.append(domain_id)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM fit_jobs WHERE {' AND '.join(clauses)} ORDER BY created_at DESC LIMIT ?",
                [*params, limit],
            ).fetchall()
        return [self._format_row(dict(r)) for r in rows]

    def save_checkpoint(
        self,
        tenant_id: str,
        domain_id: str,
        stage_name: str,
        config_hash: str,
        data: dict[str, Any],
        *,
        job_id: str | None = None,
        owner_id: str | None = None,
    ) -> str:
        key = hashlib.sha256(
            f"{tenant_id}:{domain_id}:{stage_name}:{config_hash}".encode()
        ).hexdigest()
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if (
                owner_id
                and not conn.execute(
                    "SELECT 1 FROM fit_jobs WHERE job_id = ? AND lease_owner = ? AND status = 'running' AND lease_expires_at > ?",
                    (job_id, owner_id, time.time()),
                ).fetchone()
            ):
                raise LeaseLostError("Fit job lease lost before checkpoint")
            conn.execute(
                """
                INSERT OR REPLACE INTO stage_checkpoints
                (checkpoint_key, tenant_id, domain_id, stage_name, config_hash, data_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    tenant_id,
                    domain_id,
                    stage_name,
                    config_hash,
                    json.dumps(data, default=str),
                    now,
                ),
            )
            conn.commit()
        return key

    def load_checkpoint(
        self,
        tenant_id: str,
        domain_id: str,
        stage_name: str,
        config_hash: str,
    ) -> dict[str, Any] | None:
        key = hashlib.sha256(
            f"{tenant_id}:{domain_id}:{stage_name}:{config_hash}".encode()
        ).hexdigest()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data_json FROM stage_checkpoints WHERE checkpoint_key = ?",
                (key,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def backlog_stats(self, tenant_id: str | None = None) -> dict[str, Any]:
        with self._connect() as conn:
            if tenant_id:
                rows = conn.execute(
                    "SELECT created_at FROM fit_jobs WHERE tenant_id = ? AND status IN ('queued', 'running')",
                    (tenant_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT created_at FROM fit_jobs WHERE status IN ('queued', 'running')"
                ).fetchall()
        if not rows:
            return {"active_jobs": 0, "oldest_age_seconds": 0.0}
        now = datetime.now(UTC)
        ages = [
            max(0.0, (now - datetime.fromisoformat(r[0])).total_seconds()) for r in rows
        ]
        return {
            "active_jobs": len(rows),
            "oldest_age_seconds": round(max(ages), 2),
        }

    @staticmethod
    def _format_row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "tenant_id": row["tenant_id"],
            "domain_id": row["domain_id"],
            "idempotency_key": row["idempotency_key"],
            "status": row["status"],
            "attempts": row.get("attempts", 0),
            "stage": row["stage"],
            "payload": json.loads(row["payload_json"] or "{}"),
            "progress": json.loads(row["progress_json"] or "{}"),
            "result": json.loads(row["result_json"])
            if row.get("result_json")
            else None,
            "error": row.get("error"),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
