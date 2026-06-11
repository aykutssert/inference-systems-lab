import json
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import psutil
from huggingface_hub import hf_hub_download

from local_inference.benchmark_runner import GenerationResult
from local_inference.benchmark_schema import BenchmarkPrompt
from local_inference.chat import ChatMessage
from local_inference.config import (
    GGUF_MODEL_FILE,
    GGUF_MODEL_ID,
    GGUF_MODEL_REVISION,
)

GIBIBYTE = 1024**3


def resolve_model_path() -> Path:
    return Path(
        hf_hub_download(
            repo_id=GGUF_MODEL_ID,
            filename=GGUF_MODEL_FILE,
            revision=GGUF_MODEL_REVISION,
            local_files_only=True,
        )
    )


def runtime_version(executable: str = "llama-server") -> str:
    result = subprocess.run(
        [executable, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip() or result.stderr.strip()
    if not output:
        raise RuntimeError("llama-server returned no version information")
    first_line = output.splitlines()[0]
    return first_line.removeprefix("version: ")


def parse_stream(
    lines: Iterable[bytes],
    *,
    started_at: float,
    clock: Callable[[], float],
    memory_gb: Callable[[], float],
) -> GenerationResult:
    first_token_at: float | None = None
    response_parts: list[str] = []
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    timings: dict[str, Any] | None = None
    peak_memory_gb = memory_gb()

    for raw_line in lines:
        peak_memory_gb = max(peak_memory_gb, memory_gb())
        line = raw_line.decode("utf-8").strip()
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ")
        if data == "[DONE]":
            break

        payload = json.loads(data)
        choices = payload.get("choices", [])
        if choices:
            choice = choices[0]
            content = choice.get("delta", {}).get("content")
            if content:
                if first_token_at is None:
                    first_token_at = clock()
                response_parts.append(content)
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
        if payload.get("usage") is not None:
            usage = payload["usage"]
        if payload.get("timings") is not None:
            timings = payload["timings"]

    response = "".join(response_parts).strip()
    if first_token_at is None or not response:
        raise RuntimeError("llama-server produced no response")
    if usage is None or timings is None:
        raise RuntimeError("llama-server response omitted metrics")

    return GenerationResult(
        response=response,
        finish_reason=finish_reason,
        time_to_first_token_seconds=first_token_at - started_at,
        prompt_tokens=int(usage["prompt_tokens"]),
        prompt_tokens_per_second=float(timings["prompt_per_second"]),
        generation_tokens=int(usage["completion_tokens"]),
        generation_tokens_per_second=float(timings["predicted_per_second"]),
        peak_memory_gb=peak_memory_gb,
    )


class LlamaCppBackend:
    name: Literal["mlx", "llama.cpp"] = "llama.cpp"
    model = f"{GGUF_MODEL_ID}/{GGUF_MODEL_FILE}"
    model_revision = GGUF_MODEL_REVISION

    def __init__(
        self,
        *,
        model_path: Path | None = None,
        executable: str = "llama-server",
        host: str = "127.0.0.1",
        port: int = 18080,
        startup_timeout_seconds: float = 30.0,
        request_timeout_seconds: float = 300.0,
        resolved_runtime_version: str | None = None,
    ) -> None:
        self.model_path = model_path or resolve_model_path()
        self.executable = executable
        self.host = host
        self.port = port
        self.startup_timeout_seconds = startup_timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.runtime_version = resolved_runtime_version or runtime_version(executable)
        self._process: subprocess.Popen[bytes] | None = None
        self._monitored_process: psutil.Process | None = None
        self._load_seconds: float | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def load(self) -> float:
        if self._process is not None:
            if self._load_seconds is None:
                raise RuntimeError("llama-server load state is inconsistent")
            return self._load_seconds

        started_at = time.perf_counter()
        self._process = subprocess.Popen(
            [
                self.executable,
                "-m",
                str(self.model_path),
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--gpu-layers",
                "all",
                "--reasoning",
                "off",
                "--temperature",
                "0",
                "--seed",
                "42",
                "--parallel",
                "1",
                "--no-ui",
                "--log-disable",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._monitored_process = psutil.Process(self._process.pid)

        try:
            self._wait_until_ready()
        except Exception:
            self.close()
            raise

        self._load_seconds = time.perf_counter() - started_at
        return self._load_seconds

    def _wait_until_ready(self) -> None:
        if self._process is None:
            raise RuntimeError("llama-server process has not started")

        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError("llama-server exited during startup")
            try:
                with urlopen(f"{self.base_url}/health", timeout=0.5) as response:
                    payload = json.load(response)
                    if response.status == 200 and payload.get("status") == "ok":
                        return
            except (HTTPError, URLError, TimeoutError):
                pass
            time.sleep(0.05)

        raise TimeoutError("llama-server did not become ready")

    def generate(self, prompt: BenchmarkPrompt) -> GenerationResult:
        messages: tuple[ChatMessage, ...] = ({"role": "user", "content": prompt.text},)
        return self.generate_chat(messages, prompt.max_tokens)

    def generate_chat(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> GenerationResult:
        if self._process is None or self._monitored_process is None:
            raise RuntimeError("Backend must be loaded before generation")

        request = Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": GGUF_MODEL_FILE,
                    "messages": list(messages),
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "seed": 42,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started_at = time.perf_counter()
        try:
            with urlopen(request, timeout=self.request_timeout_seconds) as response:
                return parse_stream(
                    response,
                    started_at=started_at,
                    clock=time.perf_counter,
                    memory_gb=self._memory_gb,
                )
        except RuntimeError:
            raise
        except (
            HTTPError,
            URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
            psutil.Error,
        ) as error:
            raise RuntimeError("llama-server request failed") from error

    def _memory_gb(self) -> float:
        if self._monitored_process is None:
            raise RuntimeError("llama-server process is not available")
        return self._monitored_process.memory_info().rss / GIBIBYTE

    def close(self) -> None:
        process = self._process
        self._process = None
        self._monitored_process = None
        self._load_seconds = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
