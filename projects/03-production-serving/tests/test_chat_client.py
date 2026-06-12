import io
import json
import urllib.error
from email.message import Message

import pytest

from production_serving.chat_client import format_http_error, parse_sse


def test_parse_sse_yields_json_events_until_done() -> None:
    lines = [
        b": keep-alive\n",
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
        b"\n",
        b'data: {"choices":[],"usage":{"total_tokens":4}}\n',
        b"data: [DONE]\n",
        b'data: {"ignored":true}\n',
    ]

    events = list(parse_sse(lines))

    assert events == [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [], "usage": {"total_tokens": 4}},
    ]


def test_format_http_error_reads_openai_error() -> None:
    error = urllib.error.HTTPError(
        "http://testserver/v1/chat/completions",
        429,
        "Too Many Requests",
        Message(),
        io.BytesIO(
            json.dumps(
                {
                    "error": {
                        "message": "The inference queue is full",
                        "code": "server_busy",
                    }
                }
            ).encode()
        ),
    )

    assert format_http_error(error) == (
        "HTTP 429: The inference queue is full (server_busy)"
    )


def test_parse_sse_rejects_invalid_json() -> None:
    with pytest.raises(json.JSONDecodeError):
        list(parse_sse([b"data: invalid\n"]))


def test_format_http_error_falls_back_for_invalid_body() -> None:
    error = urllib.error.HTTPError(
        "http://testserver/v1/chat/completions",
        503,
        "Service Unavailable",
        Message(),
        io.BytesIO(b'{"error": "invalid"}'),
    )

    assert format_http_error(error) == "HTTP 503: Service Unavailable"
