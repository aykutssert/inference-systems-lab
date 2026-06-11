import json
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import URLError

import pytest

from local_inference.llama_backend import (
    LlamaCppBackend,
    parse_stream,
    runtime_version,
)


def stream_chunk(payload: dict[str, object]) -> bytes:
    return f"data: {json.dumps(payload)}\n".encode()


@pytest.mark.parametrize(
    ("stdout", "stderr"),
    [
        ("version: 9590 (commit)\n", ""),
        ("", "version: 9590 (commit)\n"),
    ],
)
def test_runtime_version_accepts_both_output_streams(
    stdout: str,
    stderr: str,
) -> None:
    with patch("local_inference.llama_backend.subprocess.run") as run:
        run.return_value.stdout = stdout
        run.return_value.stderr = stderr

        assert runtime_version() == "9590 (commit)"


def test_parse_stream_maps_response_and_metrics() -> None:
    clock_values = iter([10.25])
    memory_values = iter([1.1, 1.2, 1.3, 1.25])
    lines = [
        stream_chunk(
            {
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": None},
                        "finish_reason": None,
                    }
                ]
            }
        ),
        stream_chunk(
            {
                "choices": [
                    {
                        "delta": {"content": "Hello"},
                        "finish_reason": None,
                    }
                ]
            }
        ),
        stream_chunk(
            {
                "choices": [
                    {
                        "delta": {"content": " world"},
                        "finish_reason": "stop",
                    }
                ]
            }
        ),
        stream_chunk(
            {
                "choices": [],
                "usage": {"prompt_tokens": 12, "completion_tokens": 2},
                "timings": {
                    "prompt_per_second": 200.0,
                    "predicted_per_second": 50.0,
                },
            }
        ),
        b"data: [DONE]\n",
    ]

    result = parse_stream(
        lines,
        started_at=10.0,
        clock=lambda: next(clock_values),
        memory_gb=lambda: next(memory_values, 1.25),
    )

    assert result.response == "Hello world"
    assert result.finish_reason == "stop"
    assert result.time_to_first_token_seconds == 0.25
    assert result.prompt_tokens == 12
    assert result.prompt_tokens_per_second == 200.0
    assert result.generation_tokens == 2
    assert result.generation_tokens_per_second == 50.0
    assert result.peak_memory_gb == 1.3


@pytest.mark.parametrize(
    "lines",
    [
        [b"data: [DONE]\n"],
        [
            stream_chunk(
                {
                    "choices": [
                        {
                            "delta": {"content": "response"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ),
            b"data: [DONE]\n",
        ],
    ],
)
def test_parse_stream_rejects_incomplete_response(lines: list[bytes]) -> None:
    with pytest.raises(RuntimeError):
        parse_stream(
            lines,
            started_at=1.0,
            clock=lambda: 1.1,
            memory_gb=lambda: 1.0,
        )


def test_backend_builds_pinned_server_configuration(tmp_path: Path) -> None:
    backend = LlamaCppBackend(
        model_path=tmp_path / "model.gguf",
        executable="custom-llama-server",
        port=19090,
        resolved_runtime_version="9590",
    )

    assert backend.name == "llama.cpp"
    assert backend.model.endswith("Qwen3-1.7B-Q8_0.gguf")
    assert backend.model_revision == "90862c4b9d2787eaed51d12237eafdfe7c5f6077"
    assert backend.runtime_version == "9590"
    assert backend.base_url == "http://127.0.0.1:19090"


def test_backend_requires_load_before_generation(tmp_path: Path) -> None:
    backend = LlamaCppBackend(
        model_path=tmp_path / "model.gguf",
        resolved_runtime_version="9590",
    )

    with pytest.raises(RuntimeError, match="loaded"):
        backend.generate_chat([{"role": "user", "content": "Hello"}], 16)


def test_backend_translates_request_failure(tmp_path: Path) -> None:
    backend = LlamaCppBackend(
        model_path=tmp_path / "model.gguf",
        resolved_runtime_version="9590",
    )
    backend._process = Mock()  # type: ignore[assignment]
    backend._monitored_process = Mock()

    with (
        patch(
            "local_inference.llama_backend.urlopen",
            side_effect=URLError("connection lost"),
        ),
        pytest.raises(RuntimeError, match="request failed"),
    ):
        backend.generate_chat([{"role": "user", "content": "Hello"}], 16)


class FakeProcess:
    def __init__(self) -> None:
        self.terminated = False
        self.waited = False

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: int) -> int:
        self.waited = True
        return 0

    def kill(self) -> None:
        raise AssertionError("kill must not be called")


def test_close_terminates_running_server(tmp_path: Path) -> None:
    backend = LlamaCppBackend(
        model_path=tmp_path / "model.gguf",
        resolved_runtime_version="9590",
    )
    process = FakeProcess()
    backend._process = process  # type: ignore[assignment]

    backend.close()

    assert process.terminated is True
    assert process.waited is True
