"""Prometheus metrics (RED per route template, plus model-level signals).

Each application instance owns its own :class:`~prometheus_client.CollectorRegistry`
(no global state, so several apps can coexist in one process, e.g. in tests).
Labels use the *route template* (``/v1/models/{model_id}``), never the raw path,
so label cardinality is bounded by the number of routes; requests that match
no route share the label ``<unmatched>`` and static files ``<static>``.

Exposed series:

* ``soh_http_requests_total{method,route,status}`` - rate and errors;
* ``soh_http_request_duration_seconds{method,route}`` - latency histogram;
* ``soh_http_requests_in_flight`` - concurrency;
* ``soh_inference_duration_seconds{model_id}`` - model evaluation latency;
* ``soh_predictions_total{model_id,in_domain}`` - prediction volume and the
  out-of-domain share (an early drift signal);
* ``soh_models_loaded`` - number of verified bundles being served.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    PlatformCollector,
    ProcessCollector,
    generate_latest,
)

__all__ = ["ApiMetrics"]

_HTTP_BUCKETS = (0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
_INFERENCE_BUCKETS = (0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5)


class ApiMetrics:
    """The metric families of one application instance."""

    content_type = CONTENT_TYPE_LATEST

    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        ProcessCollector(registry=self.registry)
        PlatformCollector(registry=self.registry)
        self.requests = Counter(
            "soh_http_requests_total",
            "HTTP requests by method, route template and status code.",
            ("method", "route", "status"),
            registry=self.registry,
        )
        self.latency = Histogram(
            "soh_http_request_duration_seconds",
            "HTTP request latency by method and route template.",
            ("method", "route"),
            buckets=_HTTP_BUCKETS,
            registry=self.registry,
        )
        self.in_flight = Gauge(
            "soh_http_requests_in_flight",
            "HTTP requests currently being served.",
            registry=self.registry,
        )
        self.inference_latency = Histogram(
            "soh_inference_duration_seconds",
            "Time to evaluate one estimate (model selection, prediction, interval, domain check).",
            ("model_id",),
            buckets=_INFERENCE_BUCKETS,
            registry=self.registry,
        )
        self.predictions = Counter(
            "soh_predictions_total",
            "SOH predictions served, by model and whether the input was in the training domain.",
            ("model_id", "in_domain"),
            registry=self.registry,
        )
        self.models_loaded = Gauge(
            "soh_models_loaded",
            "Verified model bundles currently served.",
            registry=self.registry,
        )

    def observe_request(self, method: str, route: str, status: int, seconds: float) -> None:
        """Record one finished HTTP request."""
        self.requests.labels(method, route, str(status)).inc()
        self.latency.labels(method, route).observe(seconds)

    def observe_prediction(self, model_id: str, in_domain: bool, seconds: float) -> None:
        """Record one served estimate."""
        self.inference_latency.labels(model_id).observe(seconds)
        self.predictions.labels(model_id, "true" if in_domain else "false").inc()

    def render(self) -> bytes:
        """Prometheus text exposition of every metric."""
        return generate_latest(self.registry)
