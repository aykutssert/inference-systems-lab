from unittest.mock import patch

from service_foundations.config import Settings
from service_foundations.main import main


def test_main_starts_uvicorn_with_settings() -> None:
    settings = Settings(host="0.0.0.0", port=9000)

    with (
        patch("service_foundations.main.get_settings", return_value=settings),
        patch("service_foundations.main.configure_logging") as configure_logging,
        patch("service_foundations.main.uvicorn.run") as run,
    ):
        main()

    configure_logging.assert_called_once_with(settings)
    run.assert_called_once()
    app = run.call_args.args[0]
    assert app.title == "service-foundations"
    assert run.call_args.kwargs == {
        "host": "0.0.0.0",
        "port": 9000,
        "log_config": None,
    }
