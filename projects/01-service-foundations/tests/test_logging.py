import json
import logging
import sys

from service_foundations.logging import JsonFormatter


def test_json_formatter_outputs_structured_log() -> None:
    formatter = JsonFormatter("test-service", "test")
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="service_ready",
        args=(),
        exc_info=None,
    )

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "info"
    assert payload["logger"] == "test.logger"
    assert payload["message"] == "service_ready"
    assert payload["service"] == "test-service"
    assert payload["environment"] == "test"
    assert payload["timestamp"].endswith("+00:00")


def test_json_formatter_includes_exception() -> None:
    formatter = JsonFormatter("test-service", "test")

    try:
        raise ValueError("invalid value")
    except ValueError:
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname=__file__,
            lineno=10,
            msg="request_failed",
            args=(),
            exc_info=sys.exc_info(),
        )

    payload = json.loads(formatter.format(record))

    assert "ValueError: invalid value" in payload["exception"]
