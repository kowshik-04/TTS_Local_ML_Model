"""Small in-process Prometheus-compatible metrics (no extra dependency)."""

from __future__ import annotations

import threading
import time
from collections import defaultdict


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._histograms: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = defaultdict(list)

    @staticmethod
    def _key(name: str, labels: dict[str, str] | None) -> tuple[str, tuple[tuple[str, str], ...]]:
        return name, tuple(sorted((labels or {}).items()))

    def inc(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            self._counters[self._key(name, labels)] += value

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            bucket = self._histograms[self._key(name, labels)]
            bucket.append(float(value))
            if len(bucket) > 2048:
                del bucket[:1024]

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            counters = dict(self._counters)
            histograms = {key: list(value) for key, value in self._histograms.items()}

        def label_text(labels: tuple[tuple[str, str], ...], extra: tuple[str, str] | None = None) -> str:
            items = list(labels)
            if extra is not None:
                items.append(extra)
            if not items:
                return ""
            encoded = ",".join(
                f'{key}="{str(value).replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
                for key, value in items
            )
            return "{" + encoded + "}"

        by_name: dict[str, list[tuple[tuple[tuple[str, str], ...], float]]] = defaultdict(list)
        for (name, labels), value in counters.items():
            by_name[name].append((labels, value))
        for name in sorted(by_name):
            lines.append(f"# TYPE {name} counter")
            for labels, value in sorted(by_name[name]):
                lines.append(f"{name}{label_text(labels)} {value:g}")

        by_name = defaultdict(list)
        for (name, labels), values in histograms.items():
            by_name[name].append((labels, values))
        for name in sorted(by_name):
            lines.append(f"# TYPE {name} summary")
            for labels, values in sorted(by_name[name]):
                if not values:
                    continue
                count = len(values)
                total = sum(values)
                sorted_values = sorted(values)
                for quantile, position in ((0.5, 0.5), (0.9, 0.9), (0.99, 0.99)):
                    index = min(count - 1, int(position * (count - 1)))
                    lines.append(
                        f"{name}{label_text(labels, ('quantile', str(quantile)))} "
                        f"{sorted_values[index]:g}"
                    )
                lines.append(f"{name}_sum{label_text(labels)} {total:g}")
                lines.append(f"{name}_count{label_text(labels)} {count:g}")
        return "\n".join(lines) + ("\n" if lines else "")


METRICS = Metrics()


class timer:
    def __init__(self, metrics: Metrics, name: str, labels: dict[str, str] | None = None) -> None:
        self.metrics = metrics
        self.name = name
        self.labels = labels
        self.started = 0.0

    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.metrics.observe(self.name, time.perf_counter() - self.started, self.labels)
        return False
