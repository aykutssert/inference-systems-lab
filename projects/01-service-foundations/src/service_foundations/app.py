import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from service_foundations.config import Settings, get_settings
from service_foundations.health import router as health_router
from service_foundations.logging import configure_logging
from service_foundations.state import ServiceState

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    service_state = ServiceState()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(resolved_settings)
        logger.info("service_starting")
        service_state.mark_ready()
        logger.info("service_ready")
        try:
            yield
        finally:
            service_state.mark_not_ready()
            logger.info("service_stopped")

    app = FastAPI(
        title=resolved_settings.service_name,
        version=resolved_settings.service_version,
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.service_state = service_state
    app.include_router(health_router)
    return app
