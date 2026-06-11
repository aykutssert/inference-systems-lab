import uvicorn

from service_foundations.app import create_app
from service_foundations.config import get_settings
from service_foundations.logging import configure_logging


def main() -> None:
    settings = get_settings()
    configure_logging(settings)
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_config=None,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
