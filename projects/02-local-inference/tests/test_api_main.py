from unittest.mock import patch

from local_inference.api_main import main


def test_main_starts_local_server() -> None:
    with (
        patch("local_inference.api_main.create_app") as create_app,
        patch("local_inference.api_main.uvicorn.run") as run,
    ):
        main()

    create_app.assert_called_once_with()
    run.assert_called_once_with(
        create_app.return_value,
        host="127.0.0.1",
        port=8000,
    )
