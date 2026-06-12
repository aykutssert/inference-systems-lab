import time
from collections.abc import MutableMapping
from typing import Any, cast

from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.types import ASGIApp, Receive, Scope, Send

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests.",
    ["method", "path", "status"],
)
REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "path"],
)
ACTIVE_REQUESTS = Gauge(
    "http_requests_active",
    "Number of HTTP requests currently being processed.",
)
TIME_TO_FIRST_TOKEN = Histogram(
    "chat_completion_time_to_first_token_seconds",
    "Time to first generated token for chat completions, in seconds.",
)
GENERATED_TOKENS = Counter(
    "chat_completion_generated_tokens_total",
    "Total number of completion tokens generated.",
)


router = APIRouter(tags=["metrics"])


@router.get("/metrics")
def metrics_endpoint() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


class MetricsMiddleware:
    """ASGI middleware that records request count, duration, and concurrency.

    Implemented at the raw ASGI layer (not BaseHTTPMiddleware) so streaming
    responses are not buffered.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope["method"]
        started_at = time.monotonic()
        status_code = 500

        async def send_wrapper(message: MutableMapping[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = cast(int, message["status"])
            await send(message)

        ACTIVE_REQUESTS.inc()
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            ACTIVE_REQUESTS.dec()
            duration = time.monotonic() - started_at
            path = route_path(scope)
            REQUEST_DURATION.labels(method=method, path=path).observe(duration)
            REQUEST_COUNT.labels(
                method=method, path=path, status=str(status_code)
            ).inc()


def route_path(scope: Scope) -> str:
    route = scope.get("route")
    if route is not None:
        return cast(str, route.path)
    return "unmatched"


def record_time_to_first_token(seconds: float) -> None:
    TIME_TO_FIRST_TOKEN.observe(seconds)


def record_generated_tokens(count: int) -> None:
    GENERATED_TOKENS.inc(count)
