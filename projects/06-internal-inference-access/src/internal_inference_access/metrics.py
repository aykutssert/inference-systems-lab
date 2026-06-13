from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter


class GatewayMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "gateway_chat_requests_total",
            "Authenticated chat completion requests.",
            ["user", "status"],
            registry=self.registry,
        )


def create_metrics_router(metrics: GatewayMetrics) -> APIRouter:
    router = APIRouter(tags=["metrics"])

    @router.get("/metrics")
    def metrics_endpoint() -> Response:
        from prometheus_client import generate_latest

        return Response(
            content=generate_latest(metrics.registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    return router
