from unittest.mock import patch

from pydantic import HttpUrl, SecretStr

from internal_inference_access.config import Settings
from internal_inference_access.main import main


def test_main_starts_gateway_with_settings() -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        api_keys={"user-a": "key-a"},
        admin_api_key=SecretStr("admin-key"),
        upstream_base_url=HttpUrl("https://inference.test"),
        upstream_api_key="upstream-key",
        host="0.0.0.0",
        port=9000,
        health_timeout_seconds=1.5,
        rate_limit_requests_per_minute=120,
        rate_limit_burst=20,
    )

    with (
        patch(
            "internal_inference_access.main.get_settings",
            return_value=settings,
        ),
        patch("internal_inference_access.main.create_app") as create_app,
        patch("internal_inference_access.main.uvicorn.run") as run,
    ):
        app = create_app.return_value
        main()

    create_app.assert_called_once_with(
        {"user-a": "key-a"},
        admin_api_key="admin-key",
        upstream_base_url="https://inference.test/",
        upstream_api_key="upstream-key",
        health_timeout_seconds=1.5,
        requests_per_minute=120,
        rate_limit_burst=20,
    )
    run.assert_called_once_with(app, host="0.0.0.0", port=9000)
