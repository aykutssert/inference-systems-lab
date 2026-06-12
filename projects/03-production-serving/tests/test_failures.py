import time
from collections.abc import Iterator, Sequence

from fastapi.testclient import TestClient
from local_inference.benchmark_runner import GenerationResult
from local_inference.chat import ChatMessage

from production_serving.app import create_app
from production_serving.streaming import GenerationChunk


def result() -> GenerationResult:
    return GenerationResult(
        response="recovered",
        finish_reason="stop",
        time_to_first_token_seconds=0.1,
        prompt_tokens=1,
        prompt_tokens_per_second=1.0,
        generation_tokens=1,
        generation_tokens_per_second=1.0,
        peak_memory_gb=1.0,
    )


class FailOnceBackend:
    model = "test-model"

    def __init__(self) -> None:
        self.calls = 0

    def load(self) -> float:
        return 0.1

    def generate_chat(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> GenerationResult:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("backend failed")
        return result()

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> Iterator[GenerationChunk]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("backend failed")
        yield GenerationChunk(text="recovered")


class TimeoutOnceBackend(FailOnceBackend):
    def generate_chat(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> GenerationResult:
        self.calls += 1
        if self.calls == 1:
            time.sleep(0.02)
        return result()


def payload(*, stream: bool = False) -> dict[str, object]:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": stream,
    }


def test_backend_failure_returns_503_and_releases_slot() -> None:
    with TestClient(
        create_app(FailOnceBackend(), max_concurrent=1, max_queued=0)
    ) as client:
        failed = client.post("/v1/chat/completions", json=payload())
        recovered = client.post("/v1/chat/completions", json=payload())

    assert failed.status_code == 503
    assert failed.json()["error"]["code"] == "backend_unavailable"
    assert recovered.status_code == 200


def test_streaming_backend_failure_returns_503_and_releases_slot() -> None:
    with TestClient(
        create_app(FailOnceBackend(), max_concurrent=1, max_queued=0)
    ) as client:
        failed = client.post("/v1/chat/completions", json=payload(stream=True))
        recovered = client.post("/v1/chat/completions", json=payload())

    assert failed.status_code == 503
    assert failed.json()["error"]["code"] == "backend_unavailable"
    assert recovered.status_code == 200


def test_timeout_returns_504_and_releases_slot() -> None:
    with TestClient(
        create_app(
            TimeoutOnceBackend(),
            timeout_seconds=0.01,
            max_concurrent=1,
            max_queued=0,
        )
    ) as client:
        timed_out = client.post("/v1/chat/completions", json=payload())
        recovered = client.post("/v1/chat/completions", json=payload())

    assert timed_out.status_code == 504
    assert timed_out.json()["error"]["code"] == "request_timeout"
    assert recovered.status_code == 200
