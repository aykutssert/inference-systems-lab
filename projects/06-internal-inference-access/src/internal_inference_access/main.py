import uvicorn

from internal_inference_access.app import create_app
from internal_inference_access.config import get_settings


def main() -> None:
    settings = get_settings()
    app = create_app(
        settings.api_keys,
        admin_api_key=settings.admin_api_key.get_secret_value(),
        upstream_base_url=str(settings.upstream_base_url),
        upstream_api_key=settings.upstream_api_key,
        health_timeout_seconds=settings.health_timeout_seconds,
        requests_per_minute=settings.rate_limit_requests_per_minute,
        rate_limit_burst=settings.rate_limit_burst,
    )
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":  # pragma: no cover
    main()
