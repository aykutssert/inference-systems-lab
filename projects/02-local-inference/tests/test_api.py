from collections.abc import Sequence

from fastapi.testclient import TestClient

from local_inference.app import create_app
from local_inference.benchmark_runner import GenerationResult
from local_inference.chat import ChatMessage


class FakeBackend:
    model = "test-model"

    def __init__(self) -> None:
        self.loaded = False
        self.messages: Sequence[ChatMessage] = ()
        self.max_tokens = 0

    def load(self) -> float:
        self.loaded = True
        return 0.1

    def generate_chat(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> GenerationResult:
        self.messages = messages
        self.max_tokens = max_tokens
        return GenerationResult(
            response="Latency is request delay.",
            finish_reason="stop",
            time_to_first_token_seconds=0.1,
            prompt_tokens=12,
            prompt_tokens_per_second=100.0,
            generation_tokens=6,
            generation_tokens_per_second=80.0,
            peak_memory_gb=1.1,
        )


class FailingBackend(FakeBackend):
    def generate_chat(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> GenerationResult:
        raise RuntimeError("sensitive backend failure")


def test_health_endpoints_reflect_service_lifecycle() -> None:
    app = create_app(FakeBackend())
    client = TestClient(app)

    before_startup = client.get("/health/ready")
    liveness = client.get("/health/live")
    client.close()

    with TestClient(app) as started_client:
        ready = started_client.get("/health/ready")

    assert before_startup.status_code == 503
    assert liveness.status_code == 200
    assert liveness.json() == {"status": "ok"}
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready"}
    assert app.state.service_state.is_ready is False


def test_lists_available_model() -> None:
    backend = FakeBackend()

    with TestClient(create_app(backend)) as client:
        response = client.get("/v1/models")

    assert backend.loaded is True
    assert response.status_code == 200
    assert response.json()["object"] == "list"
    assert response.json()["data"][0]["id"] == "test-model"


def test_creates_chat_completion() -> None:
    backend = FakeBackend()

    with TestClient(create_app(backend)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "developer", "content": "Answer briefly."},
                    {"role": "user", "content": "Define latency."},
                ],
                "max_completion_tokens": 32,
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["id"].startswith("chatcmpl-")
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "test-model"
    assert payload["choices"] == [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Latency is request delay.",
                "refusal": None,
                "annotations": [],
            },
            "logprobs": None,
            "finish_reason": "stop",
        }
    ]
    assert payload["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 6,
        "total_tokens": 18,
    }
    assert backend.messages == (
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "Define latency."},
    )
    assert backend.max_tokens == 32


def test_rejects_unknown_model() -> None:
    with TestClient(create_app(FakeBackend())) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "missing-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "message": "Model 'missing-model' is not available",
            "type": "invalid_request_error",
            "param": "model",
            "code": "model_not_found",
        }
    }


def test_rejects_streaming_and_multiple_choices() -> None:
    with TestClient(create_app(FakeBackend())) as client:
        streaming = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
        multiple_choices = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "n": 2,
            },
        )

    assert streaming.status_code == 400
    assert multiple_choices.status_code == 400


def test_rejects_empty_messages_and_unknown_fields() -> None:
    with TestClient(create_app(FakeBackend())) as client:
        empty_messages = client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": []},
        )
        unknown_field = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.5,
            },
        )

    assert empty_messages.status_code == 400
    assert unknown_field.status_code == 400


def test_supports_deprecated_max_tokens() -> None:
    backend = FakeBackend()

    with TestClient(create_app(backend)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 64,
            },
        )

    assert response.status_code == 200
    assert backend.max_tokens == 64


def test_rejects_conflicting_token_limits() -> None:
    with TestClient(create_app(FakeBackend())) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_completion_tokens": 32,
                "max_tokens": 64,
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_translates_backend_failure_without_leaking_details() -> None:
    with TestClient(create_app(FailingBackend())) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "message": "The inference backend is temporarily unavailable",
            "type": "server_error",
            "param": None,
            "code": "backend_unavailable",
        }
    }
    assert "sensitive backend failure" not in response.text
