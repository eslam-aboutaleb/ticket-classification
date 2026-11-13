"""In-process operational telemetry, stage latency percentiles, and drift/coverage counters."""

from __future__ import annotations

import resource
import sys
from collections import defaultdict, deque
from threading import RLock
from typing import Any

import numpy as np


class TelemetryCollector:
    """Thread-safe collector for stage latencies, provider retries, fallbacks, and routing coverage."""

    def __init__(self, max_samples_per_path: int = 2000) -> None:
        self._lock = RLock()
        self.max_samples = max_samples_per_path
        self._counters: dict[str, int] = defaultdict(int)
        self._labeled_counters: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._latencies_ms: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=max_samples_per_path)
        )
        self._active_versions: dict[str, str] = {}

    def inc(self, name: str, amount: int = 1, label: str | None = None) -> None:
        with self._lock:
            self._counters[name] += amount
            if label is not None:
                self._labeled_counters[name][label] += amount

    def observe_latency(self, path_name: str, elapsed_ms: float) -> None:
        with self._lock:
            self._latencies_ms[path_name].append(float(elapsed_ms))

    def set_active_version(self, tenant_id: str, domain_id: str, version: str) -> None:
        with self._lock:
            self._active_versions[f"{tenant_id}:{domain_id}"] = version

    @staticmethod
    def peak_rss_mb() -> float:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports ru_maxrss in bytes; Linux reports kilobytes.
        divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
        return round(float(usage) / divisor, 2)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latency_summary: dict[str, dict[str, float | int]] = {}
            for path_name, samples in self._latencies_ms.items():
                if not samples:
                    continue
                arr = np.asarray(samples, dtype=float)
                p50, p95, p99 = np.percentile(arr, [50, 95, 99])
                latency_summary[path_name] = {
                    "count": len(arr),
                    "p50_ms": round(float(p50), 3),
                    "p95_ms": round(float(p95), 3),
                    "p99_ms": round(float(p99), 3),
                    "mean_ms": round(float(arr.mean()), 3),
                }
            return {
                "counters": dict(self._counters),
                "by_dimension": {k: dict(v) for k, v in self._labeled_counters.items()},
                "latencies_ms": latency_summary,
                "active_model_versions": dict(self._active_versions),
                "peak_rss_mb": self.peak_rss_mb(),
            }
