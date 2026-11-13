"""Container health probes without application or model initialization."""

import argparse
import os
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=["api", "worker"])
    args = parser.parse_args(argv)
    try:
        if args.role == "api":
            with urlopen("http://127.0.0.1:8000/health", timeout=3) as response:
                return 0 if response.status == 200 else 1
        path = Path(os.environ.get("WORKER_HEALTH_FILE", "/tmp/worker-heartbeat"))
        return 0 if 0 <= time.time() - path.stat().st_mtime < 90 else 1
    except (OSError, URLError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
