import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from local_inference.api import OpenAIAPIError
from local_inference.api_models import ErrorDetail, ErrorResponse
from local_inference.health import router as health_router
from local_inference.service_state import ServiceState

from production_serving.api import StreamingBackend, router
from production_serving.config import first_token_timeout_seconds


def create_app(
    backend: StreamingBackend | None = None,
    *,
    timeout_seconds: float | None = None,
) -> FastAPI:
    if backend is None:
        from production_serving.backend import StreamingMlxBackend

        resolved_backend: StreamingBackend = StreamingMlxBackend()
    else:
        resolved_backend = backend
    service_state = ServiceState()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved_backend.load()
        service_state.mark_ready()
        app.state.started_at = int(time.time())
        try:
            yield
        finally:
            service_state.mark_not_ready()

    app = FastAPI(
        title="Production Serving",
        version="0.3.0",
        lifespan=lifespan,
    )
    app.state.backend = resolved_backend
    app.state.service_state = service_state
    app.state.started_at = int(time.time())
    app.state.first_token_timeout_seconds = (
        timeout_seconds
        if timeout_seconds is not None
        else first_token_timeout_seconds()
    )
    app.include_router(health_router)
    app.include_router(router)

    @app.exception_handler(OpenAIAPIError)
    async def openai_error_handler(
        request: object,
        error: OpenAIAPIError,
    ) -> JSONResponse:
        response = ErrorResponse(
            error=ErrorDetail(
                message=error.message,
                type=error.error_type,
                param=error.param,
                code=error.code,
            )
        )
        return JSONResponse(
            status_code=error.status_code,
            content=response.model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: object,
        error: RequestValidationError,
    ) -> JSONResponse:
        first_error = error.errors()[0]
        location = first_error.get("loc", ())
        param = str(location[-1]) if location else None
        response = ErrorResponse(
            error=ErrorDetail(
                message=str(first_error["msg"]),
                type="invalid_request_error",
                param=param,
                code=None,
            )
        )
        return JSONResponse(status_code=400, content=response.model_dump())

    return app
