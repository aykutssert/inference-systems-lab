import io
from unittest.mock import patch

import httpx
import pytest

from internal_inference_access.chat_client import main, parse_sse, stream_prompt


def test_parse_sse_stops_at_done() -> None:
    events = list(
        parse_sse(
            [
                'data: {"choices":[]}',
                "data: [DONE]",
                'data: {"ignored":true}',
            ]
        )
    )

    assert events == [{"choices": []}]


def test_stream_prompt_writes_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handle_request(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer user-key"
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=(
                b'data: {"choices":[{"delta":{"content":"hello "}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"world"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
            request=request,
        )

    original_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(handle_request),
            **kwargs,
        ),
    )
    output = io.StringIO()

    response = stream_prompt(
        "hello",
        api_key="user-key",
        base_url="https://gateway.test",
        model="test-model",
        max_tokens=16,
        output=output,
    )

    assert response == "hello world"
    assert output.getvalue() == "hello world\n"


def test_main_streams_prompt_and_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERENCE_API_KEY", "user-key")

    with (
        patch("builtins.input", side_effect=["hello", ""]),
        patch(
            "internal_inference_access.chat_client.stream_prompt",
            return_value="response",
        ) as stream,
    ):
        main()

    stream.assert_called_once()


def test_main_handles_http_error_and_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("INFERENCE_API_KEY", "user-key")
    request = httpx.Request("POST", "https://gateway.test")
    response = httpx.Response(503, request=request)

    with (
        patch("builtins.input", side_effect=["hello", KeyboardInterrupt]),
        patch(
            "internal_inference_access.chat_client.stream_prompt",
            side_effect=httpx.HTTPStatusError(
                "unavailable",
                request=request,
                response=response,
            ),
        ),
    ):
        main()

    assert "Error: unavailable" in capsys.readouterr().out


def test_main_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)

    with pytest.raises(SystemExit):
        main()
