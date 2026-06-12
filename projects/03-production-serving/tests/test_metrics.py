import re
from collections.abc import Iterator, Sequence

from fastapi.testclient import TestClient
from local_inference.benchmark_runner import GenerationResult
from local_inference.chat import ChatMessage

from production_serving.app import create_app
from production_serving.streaming import GenerationChunk


def metric_value(body: str, name: str, labels: str = "") -> float:
    pattern = rf"^{re.escape(name)}{re.escape(labels)} ([0-9.e+-]+)$"
    match = re.search(pattern, body, re.MULTILINE)
    assert match is not None, f"{name}{labels} not found in:\n{body}"
    return float(match.group(1))


class FakeBackend:
    model = "test-model"

    def load(self) -> float:
        return 0.1

    def generate_chat(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> GenerationResult:
        return GenerationResult(
            response="non-streaming response",
            finish_reason="stop",
            time_to_first_token_seconds=0.1,
            prompt_tokens=8,
            prompt_tokens_per_second=100.0,
            generation_tokens=4,
            generation_tokens_per_second=80.0,
            peak_memory_gb=1.1,
        )

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> Iterator[GenerationChunk]:
        yield GenerationChunk(text="streaming ")
        yield GenerationChunk(text="response")
        yield GenerationChunk(
            text="",
            finish_reason="stop",
            prompt_tokens=8,
            generation_tokens=4,
        )


def test_metrics_endpoint_exposes_prometheus_format() -> None:
    with TestClient(create_app(FakeBackend())) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "http_requests_total" in body
    assert "http_request_duration_seconds" in body
    assert "http_requests_active" in body
    assert "chat_completion_time_to_first_token_seconds" in body
    assert "chat_completion_generated_tokens_total" in body


def test_http_request_metrics_track_path_and_status() -> None:
    with TestClient(create_app(FakeBackend())) as client:
        client.get("/health/live")
        response = client.get("/metrics")

    body = response.text
    assert (
        metric_value(
            body,
            "http_requests_total",
            '{method="GET",path="/health/live",status="200"}',
        )
        >= 1.0
    )
    assert (
        metric_value(
            body,
            "http_request_duration_seconds_count",
            '{method="GET",path="/health/live"}',
        )
        >= 1.0
    )
    assert metric_value(body, "http_requests_active") >= 0.0


def test_unmatched_paths_use_bounded_metric_label() -> None:
    with TestClient(create_app(FakeBackend())) as client:
        client.get("/random-user-controlled-path")
        body = client.get("/metrics").text

    assert (
        metric_value(
            body,
            "http_requests_total",
            '{method="GET",path="unmatched",status="404"}',
        )
        >= 1.0
    )
    assert 'path="/random-user-controlled-path"' not in body


def test_non_streaming_completion_records_ttft_and_generated_tokens() -> None:
    with TestClient(create_app(FakeBackend())) as client:
        before = metric_value(
            client.get("/metrics").text, "chat_completion_generated_tokens_total"
        )
        client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        body = client.get("/metrics").text

    assert (
        metric_value(body, "chat_completion_time_to_first_token_seconds_count") >= 1.0
    )
    assert metric_value(body, "chat_completion_generated_tokens_total") == before + 4.0


def test_streaming_completion_records_ttft_and_generated_tokens() -> None:
    with TestClient(create_app(FakeBackend())) as client:
        before = metric_value(
            client.get("/metrics").text, "chat_completion_generated_tokens_total"
        )
        client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
        body = client.get("/metrics").text

    assert (
        metric_value(body, "chat_completion_time_to_first_token_seconds_count") >= 1.0
    )
    assert metric_value(body, "chat_completion_generated_tokens_total") == before + 4.0
