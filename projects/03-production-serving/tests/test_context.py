from collections.abc import Iterator, Sequence
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from local_inference.benchmark_runner import GenerationResult
from local_inference.chat import ChatMessage

from production_serving.app import create_app
from production_serving.backend import StreamingMlxBackend
from production_serving.context import ContextWindowExceededError
from production_serving.streaming import GenerationChunk


class ContextBackend:
    model = "test-model"

    def load(self) -> float:
        return 0.1

    def validate_context(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> None:
        if len(messages) > 1:
            raise ContextWindowExceededError(
                prompt_tokens=40_000,
                max_tokens=max_tokens,
                context_window=40_960,
            )

    def generate_chat(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> GenerationResult:
        return GenerationResult(
            response="accepted",
            finish_reason="stop",
            time_to_first_token_seconds=0.1,
            prompt_tokens=1,
            prompt_tokens_per_second=1.0,
            generation_tokens=1,
            generation_tokens_per_second=1.0,
            peak_memory_gb=1.0,
        )

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> Iterator[GenerationChunk]:
        yield GenerationChunk(text="accepted")


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages: object,
        **kwargs: object,
    ) -> str:
        return "rendered prompt"

    def encode(
        self,
        prompt: str,
        *,
        add_special_tokens: bool,
    ) -> list[int]:
        return list(range(40_000))


def test_mlx_context_validation_uses_loaded_model_limit() -> None:
    backend = StreamingMlxBackend()
    backend._model = SimpleNamespace(
        args=SimpleNamespace(max_position_embeddings=40_960)
    )
    backend._tokenizer = FakeTokenizer()

    with pytest.raises(ContextWindowExceededError) as captured:
        backend.validate_context(
            [{"role": "user", "content": "Hello"}],
            max_tokens=1024,
        )

    assert captured.value.prompt_tokens == 40_000
    assert captured.value.context_window == 40_960


def test_context_window_rejection_is_openai_compatible() -> None:
    with TestClient(create_app(ContextBackend())) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "First"},
                    {"role": "assistant", "content": "Second"},
                ],
                "max_completion_tokens": 1024,
            },
        )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": (
            "Prompt uses 40000 tokens and reserves 1024 completion tokens, "
            "exceeding the 40960-token context window"
        ),
        "type": "invalid_request_error",
        "param": "messages",
        "code": "context_length_exceeded",
    }


def test_context_validation_allows_request_within_budget() -> None:
    with TestClient(create_app(ContextBackend())) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == 200
