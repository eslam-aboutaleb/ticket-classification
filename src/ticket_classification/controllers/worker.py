"""Run lease-based fitting separately from the API: python -m ticket_classification.controllers.worker."""

import argparse
import logging.config
import os
import signal
from pathlib import Path
from threading import Event

from ticket_classification.config.logging import logging_config
from ticket_classification.config.settings import AppSettings
from ticket_classification.controllers.api import ServiceContainer
from ticket_classification.services.fit_worker import FitWorker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--lease-seconds", type=float, default=60)
    parser.add_argument("--poll-seconds", type=float, default=1)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if args.lease_seconds <= 0 or args.poll_seconds <= 0:
        parser.error("lease and poll durations must be positive")
    logging.config.dictConfig(logging_config())
    stop = Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    overrides = {
        key: value
        for key, value in {
            "config_path": args.config,
            "data_dir": args.data_dir,
        }.items()
        if value is not None
    }
    container = ServiceContainer(settings=AppSettings(**overrides))
    try:
        health_path = os.environ.get("WORKER_HEALTH_FILE")
        worker = FitWorker(
            container,
            lease_seconds=args.lease_seconds,
            health_file=Path(health_path) if health_path else None,
        )
        if args.once:
            worker.run_once()
        else:
            worker.run_forever(stop, args.poll_seconds)
    finally:
        container.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
