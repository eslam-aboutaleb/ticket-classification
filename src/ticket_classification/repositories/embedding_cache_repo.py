"""SQLite Proxy-Cache Repository for Cloud Embeddings with tenant scoping, WAL mode, and single-flight leases."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path


class EmbeddingCacheRepository:
    """Content-addressable SQLite store keyed by (tenant_id, embedding_identity, text_hash)."""

    def __init__(
        self,
        db_path: str | Path = "data/cache/embeddings_cache.sqlite",
        busy_timeout_ms: int = 5000,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.busy_timeout_ms = busy_timeout_ms
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=self.busy_timeout_ms / 1000.0)
        conn.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    embedding_identity TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    PRIMARY KEY (tenant_id, embedding_identity, text_hash)
                )
                """
            )
            columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(embedding_cache)").fetchall()
            }
            if "tenant_id" not in columns:
                conn.execute(
                    "ALTER TABLE embedding_cache RENAME TO embedding_cache_legacy"
                )
                conn.execute(
                    """
                    CREATE TABLE embedding_cache (
                        tenant_id TEXT NOT NULL DEFAULT 'default',
                        embedding_identity TEXT NOT NULL,
                        text_hash TEXT NOT NULL,
                        vector_json TEXT NOT NULL,
                        dimensions INTEGER NOT NULL,
                        PRIMARY KEY (tenant_id, embedding_identity, text_hash)
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO embedding_cache
                    (tenant_id, embedding_identity, text_hash, vector_json, dimensions)
                    SELECT 'default', embedding_identity, text_hash, vector_json, dimensions
                    FROM embedding_cache_legacy
                    """
                )
                conn.execute("DROP TABLE embedding_cache_legacy")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_leases (
                    tenant_id TEXT NOT NULL,
                    embedding_identity TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    PRIMARY KEY (tenant_id, embedding_identity, text_hash)
                )
                """
            )
            conn.commit()

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_many(
        self,
        embedding_identity: str,
        texts: list[str],
        tenant_id: str = "default",
    ) -> dict[str, list[float]]:
        """Retrieve cached vectors keyed by text_hash for the given (tenant_id, embedding_identity)."""
        if not texts:
            return {}
        hashes = list({self.hash_text(t) for t in texts})
        results: dict[str, list[float]] = {}
        chunk_size = 500
        with self._connect() as conn:
            for i in range(0, len(hashes), chunk_size):
                batch = hashes[i : i + chunk_size]
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(
                    f"""
                    SELECT text_hash, vector_json
                    FROM embedding_cache
                    WHERE tenant_id = ? AND embedding_identity = ? AND text_hash IN ({placeholders})
                    """,
                    [tenant_id, embedding_identity, *batch],
                ).fetchall()
                for text_hash, vec_json in rows:
                    results[text_hash] = json.loads(vec_json)
        return results

    def put_many(
        self,
        embedding_identity: str,
        text_to_vector: dict[str, list[float]],
        tenant_id: str = "default",
    ) -> None:
        """Persist a batch of vectors atomically and clear corresponding leases."""
        if not text_to_vector:
            return
        rows = [
            (
                tenant_id,
                embedding_identity,
                self.hash_text(text),
                json.dumps(vec),
                len(vec),
            )
            for text, vec in text_to_vector.items()
        ]
        hashes = [r[2] for r in rows]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embedding_cache
                (tenant_id, embedding_identity, text_hash, vector_json, dimensions)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            chunk_size = 500
            for i in range(0, len(hashes), chunk_size):
                batch = hashes[i : i + chunk_size]
                placeholders = ",".join("?" for _ in batch)
                conn.execute(
                    f"""
                    DELETE FROM embedding_leases
                    WHERE tenant_id = ? AND embedding_identity = ? AND text_hash IN ({placeholders})
                    """,
                    [tenant_id, embedding_identity, *batch],
                )
            conn.commit()

    def acquire_leases(
        self,
        embedding_identity: str,
        texts: list[str],
        owner_id: str,
        ttl_seconds: float = 30.0,
        tenant_id: str = "default",
    ) -> list[str]:
        """Acquire single-flight leases for uncached texts; returns subset owned by owner_id."""
        if not texts:
            return []
        now = time.time()
        expires = now + ttl_seconds
        acquired: list[str] = []
        with self._connect() as conn:
            conn.execute("DELETE FROM embedding_leases WHERE expires_at < ?", (now,))
            for text in dict.fromkeys(texts):
                th = self.hash_text(text)
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO embedding_leases
                    (tenant_id, embedding_identity, text_hash, owner_id, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (tenant_id, embedding_identity, th, owner_id, expires),
                )
                if cur.rowcount > 0:
                    acquired.append(text)
                else:
                    row = conn.execute(
                        """
                        SELECT owner_id FROM embedding_leases
                        WHERE tenant_id = ? AND embedding_identity = ? AND text_hash = ?
                        """,
                        (tenant_id, embedding_identity, th),
                    ).fetchone()
                    if row and row[0] == owner_id:
                        acquired.append(text)
            conn.commit()
        return acquired
