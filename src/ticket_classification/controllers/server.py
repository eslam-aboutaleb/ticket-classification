"""Run the HTTP API with explicit process and connection limits."""

import argparse

import uvicorn

from ticket_classification.config.logging import logging_config
from ticket_classification.config.settings import AppSettings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit-concurrency", type=int, default=64)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.limit_concurrency < 1 or not 1 <= args.port <= 65535:
        parser.error("workers, concurrency and port must be positive and valid")
    AppSettings()
    uvicorn.run(
        "ticket_classification.controllers.api:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
        limit_concurrency=args.limit_concurrency,
        timeout_keep_alive=5,
        timeout_graceful_shutdown=30,
        server_header=False,
        proxy_headers=False,
        access_log=False,
        log_config=logging_config(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
