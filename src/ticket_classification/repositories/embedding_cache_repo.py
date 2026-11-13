"""SQLite Proxy-Cache Repository for Cloud Embeddings with embedding_identity validation."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


class EmbeddingCacheRepository:
    """Content-addressable SQLite store keyed by (embedding_identity, text_sha256)."""

    def __init__(
        self, db_path: str | Path = "data/cache/embeddings_cache.sqlite"
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    embedding_identity TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    PRIMARY KEY (embedding_identity, text_hash)
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
    ) -> dict[str, list[float]]:
        """Retrieve cached vectors keyed by text_hash for the given embedding_identity."""
        if not texts:
            return {}
        hashes = list({self.hash_text(t) for t in texts})
        results: dict[str, list[float]] = {}
        chunk_size = 500
        with sqlite3.connect(self.db_path) as conn:
            for i in range(0, len(hashes), chunk_size):
                batch = hashes[i : i + chunk_size]
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(
                    f"""
                    SELECT text_hash, vector_json
                    FROM embedding_cache
                    WHERE embedding_identity = ? AND text_hash IN ({placeholders})
                    """,
                    [embedding_identity, *batch],
                ).fetchall()
                for text_hash, vec_json in rows:
                    results[text_hash] = json.loads(vec_json)
        return results

    def put_many(
        self,
        embedding_identity: str,
        text_to_vector: dict[str, list[float]],
    ) -> None:
        """Persist a batch of vectors atomically."""
        if not text_to_vector:
            return
        rows = [
            (
                embedding_identity,
                self.hash_text(text),
                json.dumps(vec),
                len(vec),
            )
            for text, vec in text_to_vector.items()
        ]
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embedding_cache
                (embedding_identity, text_hash, vector_json, dimensions)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
