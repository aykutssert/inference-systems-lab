from production_serving.failure_runner import has_content_token


def test_has_content_token_ignores_role_and_usage_events() -> None:
    events = [
        {"choices": [{"delta": {"role": "assistant", "content": ""}}]},
        {"choices": [], "usage": {"completion_tokens": 4}},
    ]

    assert has_content_token(events) is False


def test_has_content_token_stops_at_content() -> None:
    events = [
        {"choices": [{"delta": {"content": "token"}}]},
        {"choices": [{"delta": {"content": "ignored"}}]},
    ]

    assert has_content_token(events) is True
