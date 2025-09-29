from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Union

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TagValue = Union[str, int, float, bool]
TagsType = Optional[Mapping[str, TagValue]]


class MetricsFacade:
    """
    Metrics facade mirroring common Datadog Python client patterns.
    Instead of sending metrics, logs them at INFO level.

    Supported patterns (logged only):
      - counters: increment, decrement
      - gauges: gauge
      - histograms: histogram
      - distributions: distribution
      - sets: set
      - timers: timing
      - events: event
      - service checks: service_check
      - increments via 'increment' alias for familiarity
    """

    def __init__(self, namespace: Optional[str] = None, default_tags: TagsType = None) -> None:
        self.namespace = namespace or "app"
        self.default_tags: Dict[str, TagValue] = dict(default_tags or {})
        logger.info("DatadogMetrics initialized namespace=%s default_tags=%s", self.namespace, self.default_tags)

    # -----------------
    # Internal helpers
    # -----------------
    def _merge_tags(self, tags: TagsType = None) -> Dict[str, TagValue]:
        merged = dict(self.default_tags)
        if tags:
            merged.update(tags)
        return merged

    def _log(self, kind: str, name: str, value: Any = None, tags: TagsType = None, **kwargs: Any) -> None:
        merged_tags = self._merge_tags(tags)
        payload = {"kind": kind, "ns": self.namespace, "name": name, "value": value, "tags": merged_tags}
        if kwargs:
            payload.update(kwargs)
        logger.info("metric %s", payload)

    # -----------------
    # Metric methods
    # -----------------
    # Counter
    def increment(self, name: str, value: float = 1.0, tags: TagsType = None, sample_rate: float = 1.0) -> None:
        self._log("counter_increment", name, value, tags, sample_rate=sample_rate)

    def decrement(self, name: str, value: float = 1.0, tags: TagsType = None, sample_rate: float = 1.0) -> None:
        self._log("counter_decrement", name, value, tags, sample_rate=sample_rate)

    # Alias commonly used
    def incr(self, name: str, value: float = 1.0, tags: TagsType = None, sample_rate: float = 1.0) -> None:
        self.increment(name, value, tags, sample_rate)

    def decr(self, name: str, value: float = 1.0, tags: TagsType = None, sample_rate: float = 1.0) -> None:
        self.decrement(name, value, tags, sample_rate)

    # Gauge
    def gauge(self, name: str, value: float, tags: TagsType = None, sample_rate: float = 1.0) -> None:
        self._log("gauge", name, value, tags, sample_rate=sample_rate)

    # Histogram
    def histogram(self, name: str, value: float, tags: TagsType = None, sample_rate: float = 1.0) -> None:
        self._log("histogram", name, value, tags, sample_rate=sample_rate)

    # Distribution
    def distribution(self, name: str, value: float, tags: TagsType = None, sample_rate: float = 1.0) -> None:
        self._log("distribution", name, value, tags, sample_rate=sample_rate)

    # Set
    def set(self, name: str, value: Union[int, float, str], tags: TagsType = None, sample_rate: float = 1.0) -> None:
        self._log("set", name, value, tags, sample_rate=sample_rate)

    # Timing (ms)
    def timing(self, name: str, ms: float, tags: TagsType = None, sample_rate: float = 1.0) -> None:
        self._log("timing", name, ms, tags, sample_rate=sample_rate)

    # -----------------
    # Events
    # -----------------
    def event(
        self,
        title: str,
        text: str,
        alert_type: Optional[str] = None,
        aggregation_key: Optional[str] = None,
        source_type_name: Optional[str] = None,
        date_happened: Optional[int] = None,
        priority: Optional[str] = None,
        hostname: Optional[str] = None,
        tags: TagsType = None,
    ) -> None:
        self._log(
            "event",
            title,
            text,
            tags,
            alert_type=alert_type,
            aggregation_key=aggregation_key,
            source_type_name=source_type_name,
            date_happened=date_happened,
            priority=priority,
            hostname=hostname,
        )

    # -----------------
    # Service checks
    # -----------------
    def service_check(
        self,
        name: str,
        status: int,
        tags: TagsType = None,
        hostname: Optional[str] = None,
        message: Optional[str] = None,
        timestamp: Optional[int] = None,
    ) -> None:
        self._log(
            "service_check",
            name,
            status,
            tags,
            hostname=hostname,
            message=message,
            timestamp=timestamp,
        )
