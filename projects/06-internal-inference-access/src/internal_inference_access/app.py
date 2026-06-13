import hmac
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, cast

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import Response

from internal_inference_access.auth import APIKeyAuthenticator, require_user
from internal_inference_access.metrics import GatewayMetrics, create_metrics_router
from internal_inference_access.proxy import InferenceProxy
from internal_inference_access.rate_limit import UserRateLimiter

logger = logging.getLogger(__name__)


def get_proxy(request: Request) -> InferenceProxy:
    return cast(InferenceProxy, request.app.state.proxy)


def create_app(
    user_keys: dict[str, str],
    *,
    admin_api_key: str,
    upstream_base_url: str = "http://127.0.0.1:8000",
    upstream_api_key: str | None = None,
    health_timeout_seconds: float = 2.0,
    requests_per_minute: int = 60,
    rate_limit_burst: int = 10,
    http_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    if admin_api_key in user_keys.values():
        raise ValueError("admin_api_key must be distinct from user API keys")
    resolved_client = http_client or httpx.AsyncClient(timeout=60.0)
    metrics = GatewayMetrics()
    rate_limiter = UserRateLimiter(requests_per_minute, rate_limit_burst)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if http_client is None:
                await resolved_client.aclose()

    app = FastAPI(
        title="Internal Inference Access",
        version="0.6.0",
        lifespan=lifespan,
    )
    app.state.authenticator = APIKeyAuthenticator(user_keys)
    app.state.proxy = InferenceProxy(
        resolved_client,
        upstream_base_url,
        upstream_api_key,
        health_timeout_seconds,
    )
    app.include_router(create_metrics_router(metrics))

    @app.get("/health/live")
    def liveness() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def readiness(
        proxy: Annotated[InferenceProxy, Depends(get_proxy)],
    ) -> dict[str, str]:
        if not await proxy.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference upstream is not ready",
            )
        return {"status": "ready"}

    @app.get("/v1/whoami")
    def whoami(user: Annotated[str, Depends(require_user)]) -> dict[str, str]:
        return {"user": user}

    def require_admin(
        x_admin_key: Annotated[str | None, Header()] = None,
    ) -> None:
        if x_admin_key is None or not hmac.compare_digest(
            x_admin_key,
            admin_api_key,
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing admin key",
            )

    @app.delete("/admin/users/{user}/access")
    def revoke_user_access(
        user: str,
        _: Annotated[None, Depends(require_admin)],
    ) -> dict[str, str]:
        if not app.state.authenticator.revoke(user):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        logger.warning(
            "gateway_user_access_revoked",
            extra={"authenticated_user": user},
        )
        return {"user": user, "status": "revoked"}

    @app.put("/admin/users/{user}/access")
    def restore_user_access(
        user: str,
        _: Annotated[None, Depends(require_admin)],
    ) -> dict[str, str]:
        if not app.state.authenticator.restore(user):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        logger.info(
            "gateway_user_access_restored",
            extra={"authenticated_user": user},
        )
        return {"user": user, "status": "active"}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        authenticated_user: Annotated[str, Depends(require_user)],
        proxy: Annotated[InferenceProxy, Depends(get_proxy)],
    ) -> Response:
        if not await rate_limiter.allow(authenticated_user):
            metrics.requests.labels(
                user=authenticated_user, status="rate_limited"
            ).inc()
            logger.warning(
                "gateway_request_rate_limited",
                extra={"authenticated_user": authenticated_user},
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )

        response = await proxy.forward(request)
        metrics.requests.labels(
            user=authenticated_user,
            status=str(response.status_code),
        ).inc()
        logger.info(
            "gateway_request_completed",
            extra={
                "authenticated_user": authenticated_user,
                "status_code": response.status_code,
            },
        )
        return response

    return app
