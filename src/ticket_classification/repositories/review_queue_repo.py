"""Persistent masked items awaiting review, adjudication, or a later discovery run."""

from __future__ import annotations

import base64
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


class ReviewQueueRepository:
    """Durable review queue with stable IDs, lifecycle states, cursor pagination, and audit trail."""

    def __init__(self, path: str | Path, busy_timeout_ms: int = 5000) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.busy_timeout_ms = busy_timeout_ms
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=self.busy_timeout_ms / 1000.0)
        conn.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS review_items (
                    review_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    domain_id TEXT NOT NULL,
                    ticket_id TEXT NOT NULL,
                    taxonomy_version TEXT NOT NULL,
                    text TEXT NOT NULL,
                    languages TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    adjudicated_label TEXT,
                    approved_for_training INTEGER NOT NULL DEFAULT 0,
                    occurrence_count INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, domain_id, ticket_id)
                )
                """
            )
            columns = {
                row[1]
                for row in db.execute("PRAGMA table_info(review_items)").fetchall()
            }
            if "review_id" not in columns:
                db.execute("ALTER TABLE review_items RENAME TO review_items_legacy")
                db.execute(
                    """
                    CREATE TABLE review_items (
                        review_id TEXT NOT NULL,
                        tenant_id TEXT NOT NULL DEFAULT 'default',
                        domain_id TEXT NOT NULL,
                        ticket_id TEXT NOT NULL,
                        taxonomy_version TEXT NOT NULL,
                        text TEXT NOT NULL,
                        languages TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        adjudicated_label TEXT,
                        approved_for_training INTEGER NOT NULL DEFAULT 0,
                        occurrence_count INTEGER NOT NULL DEFAULT 1,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (tenant_id, domain_id, ticket_id)
                    )
                    """
                )
                rows = db.execute("SELECT * FROM review_items_legacy").fetchall()
                now = datetime.now(UTC).isoformat()
                for row in rows:
                    db.execute(
                        """
                        INSERT OR REPLACE INTO review_items
                        (review_id, tenant_id, domain_id, ticket_id, taxonomy_version, text, languages, reason, status, approved_for_training, occurrence_count, created_at, updated_at)
                        VALUES (?, 'default', ?, ?, ?, ?, ?, ?, 'pending', 0, 1, ?, ?)
                        """,
                        (
                            uuid4().hex,
                            row[0],
                            row[1],
                            row[2],
                            row[3],
                            row[4],
                            row[5],
                            row[6] if len(row) > 6 and row[6] else now,
                            now,
                        ),
                    )
                db.execute("DROP TABLE review_items_legacy")
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_review_items_lookup ON review_items (tenant_id, domain_id, status, created_at, ticket_id)"
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS review_audit_events (
                    event_id TEXT PRIMARY KEY,
                    review_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    domain_id TEXT NOT NULL,
                    ticket_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            db.commit()

    def put(
        self,
        domain_id: str,
        ticket_id: str,
        version: str,
        text: str,
        languages: list[str],
        reason: str,
        tenant_id: str = "default",
    ) -> str:
        """Upsert review item by (tenant_id, domain_id, ticket_id) without duplicating across taxonomy versions."""
        now = datetime.now(UTC).isoformat()
        with self._connect() as db:
            existing = db.execute(
                """
                SELECT review_id, occurrence_count, status, created_at
                FROM review_items
                WHERE tenant_id = ? AND domain_id = ? AND ticket_id = ?
                """,
                (tenant_id, domain_id, ticket_id),
            ).fetchone()
            if existing:
                review_id, count, status, created_at = existing
                new_status = "pending" if status == "resolved" else status
                db.execute(
                    """
                    UPDATE review_items
                    SET taxonomy_version = ?, text = ?, languages = ?, reason = ?,
                        status = ?, occurrence_count = ?, updated_at = ?
                    WHERE tenant_id = ? AND domain_id = ? AND ticket_id = ?
                    """,
                    (
                        version,
                        text,
                        json.dumps(languages),
                        reason,
                        new_status,
                        int(count) + 1,
                        now,
                        tenant_id,
                        domain_id,
                        ticket_id,
                    ),
                )
            else:
                review_id = uuid4().hex
                created_at = now
                db.execute(
                    """
                    INSERT INTO review_items
                    (review_id, tenant_id, domain_id, ticket_id, taxonomy_version, text, languages, reason, status, approved_for_training, occurrence_count, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, 1, ?, ?)
                    """,
                    (
                        review_id,
                        tenant_id,
                        domain_id,
                        ticket_id,
                        version,
                        text,
                        json.dumps(languages),
                        reason,
                        created_at,
                        now,
                    ),
                )
            db.execute(
                """
                INSERT INTO review_audit_events
                (event_id, review_id, tenant_id, domain_id, ticket_id, action, actor, details_json, created_at)
                VALUES (?, ?, ?, ?, ?, 'enqueued', 'system', ?, ?)
                """,
                (
                    uuid4().hex,
                    review_id,
                    tenant_id,
                    domain_id,
                    ticket_id,
                    json.dumps({"taxonomy_version": version, "reason": reason}),
                    now,
                ),
            )
            db.commit()
            return str(review_id)

    def adjudicate(
        self,
        domain_id: str,
        ticket_id: str,
        status: str,
        adjudicated_label: str | None = None,
        approved_for_training: bool = False,
        reviewer: str = "operator",
        notes: str = "",
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        """Record human adjudication and consent for training incorporation."""
        if approved_for_training and status not in {"approved", "reviewed"}:
            raise ValueError(
                "Only approved or reviewed items can be marked approved_for_training"
            )
        now = datetime.now(UTC).isoformat()
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            row = db.execute(
                """
                SELECT * FROM review_items
                WHERE tenant_id = ? AND domain_id = ? AND ticket_id = ?
                """,
                (tenant_id, domain_id, ticket_id),
            ).fetchone()
            if not row:
                raise KeyError(f"Review item not found: {ticket_id}")
            review_id = row["review_id"]
            db.execute(
                """
                UPDATE review_items
                SET status = ?, adjudicated_label = ?, approved_for_training = ?, updated_at = ?
                WHERE tenant_id = ? AND domain_id = ? AND ticket_id = ?
                """,
                (
                    status,
                    adjudicated_label,
                    1 if approved_for_training else 0,
                    now,
                    tenant_id,
                    domain_id,
                    ticket_id,
                ),
            )
            db.execute(
                """
                INSERT INTO review_audit_events
                (event_id, review_id, tenant_id, domain_id, ticket_id, action, actor, details_json, created_at)
                VALUES (?, ?, ?, ?, ?, 'adjudicated', ?, ?, ?)
                """,
                (
                    uuid4().hex,
                    review_id,
                    tenant_id,
                    domain_id,
                    ticket_id,
                    reviewer,
                    json.dumps(
                        {
                            "status": status,
                            "adjudicated_label": adjudicated_label,
                            "approved_for_training": approved_for_training,
                            "notes": notes,
                        }
                    ),
                    now,
                ),
            )
            updated = db.execute(
                """
                SELECT * FROM review_items
                WHERE tenant_id = ? AND domain_id = ? AND ticket_id = ?
                """,
                (tenant_id, domain_id, ticket_id),
            ).fetchone()
            db.commit()
        return self._format_row(dict(updated))

    def list_items(
        self,
        domain_id: str,
        limit: int = 100,
        cursor: str | None = None,
        status: str | None = None,
        tenant_id: str = "default",
    ) -> list[dict[str, Any]]:
        """Return review items ordered by (created_at, ticket_id) with optional cursor and status filter."""
        page, _ = self.list_page(
            domain_id=domain_id,
            limit=limit,
            cursor=cursor,
            status=status,
            tenant_id=tenant_id,
        )
        return page

    def list_page(
        self,
        domain_id: str,
        limit: int = 100,
        cursor: str | None = None,
        status: str | None = None,
        tenant_id: str = "default",
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Cursor-paginated query returning (items, next_cursor)."""
        clauses = ["tenant_id = ?", "domain_id = ?"]
        params: list[Any] = [tenant_id, domain_id]
        if status:
            clauses.append("status = ?")
            params.append(status)
        if cursor:
            decoded_cursor = cursor
            if "|" not in cursor:
                try:
                    padded = cursor + "=" * (-len(cursor) % 4)
                    decoded_cursor = base64.urlsafe_b64decode(
                        padded.encode("ascii")
                    ).decode("utf-8")
                except (ValueError, UnicodeDecodeError):
                    decoded_cursor = cursor
            if "|" in decoded_cursor:
                c_time, c_tid = decoded_cursor.split("|", 1)
                clauses.append("(created_at > ? OR (created_at = ? AND ticket_id > ?))")
                params.extend([c_time, c_time, c_tid])
        where_sql = " AND ".join(clauses)
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                f"SELECT * FROM review_items WHERE {where_sql} ORDER BY created_at, ticket_id LIMIT ?",
                [*params, limit + 1],
            ).fetchall()
        has_more = len(rows) > limit
        sliced = rows[:limit]
        items = [self._format_row(dict(r)) for r in sliced]
        raw_cursor = (
            f"{items[-1]['created_at']}|{items[-1]['ticket_id']}"
            if has_more and items
            else None
        )
        next_cursor = (
            base64.urlsafe_b64encode(raw_cursor.encode("utf-8")).decode("ascii")
            if raw_cursor
            else None
        )
        return items, next_cursor

    def list_approved_training_items(
        self, domain_id: str, tenant_id: str = "default"
    ) -> list[dict[str, Any]]:
        """Return only human-approved items explicitly consented for taxonomy training."""
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                """
                SELECT * FROM review_items
                WHERE tenant_id = ? AND domain_id = ?
                  AND approved_for_training = 1
                  AND status IN ('approved', 'reviewed')
                  AND adjudicated_label IS NOT NULL
                ORDER BY created_at, ticket_id
                """,
                (tenant_id, domain_id),
            ).fetchall()
        return [self._format_row(dict(r)) for r in rows]

    def list_audit_events(
        self, domain_id: str, ticket_id: str | None = None, tenant_id: str = "default"
    ) -> list[dict[str, Any]]:
        clauses = ["tenant_id = ?", "domain_id = ?"]
        params: list[Any] = [tenant_id, domain_id]
        if ticket_id:
            clauses.append("ticket_id = ?")
            params.append(ticket_id)
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                f"SELECT * FROM review_audit_events WHERE {' AND '.join(clauses)} ORDER BY created_at",
                params,
            ).fetchall()
        return [{**dict(r), "details": json.loads(r["details_json"])} for r in rows]

    def count_backlog(self, domain_id: str, tenant_id: str = "default") -> int:
        with self._connect() as db:
            row = db.execute(
                "SELECT COUNT(*) FROM review_items WHERE tenant_id = ? AND domain_id = ? AND status = 'pending'",
                (tenant_id, domain_id),
            ).fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def _format_row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            **row,
            "languages": json.loads(row["languages"]),
            "approved_for_training": bool(row.get("approved_for_training", 0)),
        }
