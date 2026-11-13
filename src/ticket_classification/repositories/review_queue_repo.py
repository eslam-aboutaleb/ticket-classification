"""Persistent masked items awaiting review or a later discovery run."""

import json
import sqlite3
from pathlib import Path


class ReviewQueueRepository:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS review_items (domain_id TEXT, ticket_id TEXT, taxonomy_version TEXT, text TEXT, languages TEXT, reason TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(domain_id, ticket_id, taxonomy_version))"
            )

    def put(
        self,
        domain_id: str,
        ticket_id: str,
        version: str,
        text: str,
        languages: list[str],
        reason: str,
    ) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT OR REPLACE INTO review_items (domain_id,ticket_id,taxonomy_version,text,languages,reason) VALUES (?,?,?,?,?,?)",
                (domain_id, ticket_id, version, text, json.dumps(languages), reason),
            )

    def list_items(self, domain_id: str, limit: int = 100) -> list[dict]:
        with sqlite3.connect(self.path) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                "SELECT * FROM review_items WHERE domain_id=? ORDER BY created_at,ticket_id LIMIT ?",
                (domain_id, limit),
            ).fetchall()
        return [
            {**dict(row), "languages": json.loads(row["languages"])} for row in rows
        ]
