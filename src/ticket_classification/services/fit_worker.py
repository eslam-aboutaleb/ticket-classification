"""Durable job consumer; run in a dedicated process, independent of HTTP serving."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from threading import Event, Thread
from uuid import uuid4

import pandas as pd

from ticket_classification.repositories.fit_job_repo import LeaseLostError

logger = logging.getLogger(__name__)


class FitWorker:
    def __init__(
        self, container, lease_seconds: float = 60, health_file: Path | None = None
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.container = container
        self.repo = container.fit_job_repo
        self.lease_seconds = lease_seconds
        self.health_file = health_file

    def heartbeat(self) -> None:
        if self.health_file is not None:
            self.health_file.touch(mode=0o600)

    def run_once(self) -> bool:
        # A unique token per attempt fences old attempts, including same-process retries.
        owner = uuid4().hex
        self.heartbeat()
        job = self.repo.claim_next(
            owner, self.lease_seconds, self.container.settings.fit_job_max_attempts
        )
        if job is None:
            return False
        job_id = job["job_id"]
        done = Event()

        def heartbeat() -> None:
            while not done.wait(self.lease_seconds / 3):
                try:
                    if not self.repo.renew_lease(job_id, owner, self.lease_seconds):
                        return
                    self.heartbeat()
                except Exception:
                    logger.exception("Fit lease renewal failed for job %s", job_id)
                    return

        pulse = Thread(target=heartbeat, name="fit-lease", daemon=True)
        pulse.start()
        service = None
        try:
            service = self.container.get_service(
                job["domain_id"], tenant_id=job["tenant_id"]
            )
            payload = job["payload"]
            metrics = service.fit_dataframe(
                pd.DataFrame(payload["records"]),
                strategy=payload.get("strategy"),
                use_llm_naming=payload.get("use_llm_naming", True),
                provider_override=payload.get("provider_override"),
                oos_texts=payload.get("oos_texts"),
                job_id=job_id,
                job_owner_id=owner,
            )
            self.repo.update_stage(
                job_id,
                "completed",
                "completed",
                result={k: v for k, v in metrics.items() if not k.startswith("_")},
                owner_id=owner,
            )
        except LeaseLostError:
            logger.info("Fit attempt fenced for job %s", job_id)
        except Exception as exc:  # noqa: BLE001 - persist arbitrary fit failures without killing the consumer
            # A stale attempt must never overwrite a replacement attempt's result.
            try:
                self.repo.update_stage(
                    job_id, "failed", "failed", error=str(exc), owner_id=owner
                )
            except LeaseLostError:
                logger.info("Failed fit attempt fenced for job %s", job_id)
        finally:
            done.set()
            pulse.join(timeout=self.lease_seconds)
            if service is not None:
                service._job_owners.pop(job_id, None)
        return True

    def run_forever(self, stop: Event, poll_seconds: float = 1) -> None:
        while not stop.is_set():
            try:
                worked = self.run_once()
            except sqlite3.OperationalError:
                logger.exception("Fit queue temporarily unavailable")
                worked = False
            if not worked:
                stop.wait(poll_seconds)
