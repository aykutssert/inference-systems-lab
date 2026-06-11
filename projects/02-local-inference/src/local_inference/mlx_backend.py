import time
from collections.abc import Sequence
from typing import Any, Literal

import mlx.core as mx
from mlx_lm import __version__ as mlx_lm_version
from mlx_lm import load, stream_generate

from local_inference.benchmark_runner import GenerationResult
from local_inference.benchmark_schema import BenchmarkPrompt
from local_inference.chat import ChatMessage, build_prompt
from local_inference.config import MODEL_ID, MODEL_REVISION


class MlxBackend:
    name: Literal["mlx", "llama.cpp"] = "mlx"
    model = MODEL_ID
    model_revision = MODEL_REVISION
    runtime_version = mlx_lm_version

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> float:
        started_at = time.perf_counter()
        loaded = load(self.model, revision=self.model_revision)
        self._model = loaded[0]
        self._tokenizer = loaded[1]
        return time.perf_counter() - started_at

    def generate(self, prompt: BenchmarkPrompt) -> GenerationResult:
        messages: tuple[ChatMessage, ...] = ({"role": "user", "content": prompt.text},)
        return self.generate_chat(messages, prompt.max_tokens)

    def generate_chat(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> GenerationResult:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Backend must be loaded before generation")

        formatted_prompt = build_prompt(self._tokenizer, messages)
        mx.reset_peak_memory()
        generation_started_at = time.perf_counter()
        first_token_at: float | None = None
        response_parts: list[str] = []
        final_response: Any = None

        for generation_response in stream_generate(
            self._model,
            self._tokenizer,
            formatted_prompt,
            max_tokens=max_tokens,
        ):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            response_parts.append(generation_response.text)
            final_response = generation_response

        if first_token_at is None or final_response is None:
            raise RuntimeError("Model produced no generation response")

        response = "".join(response_parts).strip()
        if not response:
            raise RuntimeError("Model produced an empty response")

        return GenerationResult(
            response=response,
            finish_reason=final_response.finish_reason,
            time_to_first_token_seconds=first_token_at - generation_started_at,
            prompt_tokens=final_response.prompt_tokens,
            prompt_tokens_per_second=final_response.prompt_tps,
            generation_tokens=final_response.generation_tokens,
            generation_tokens_per_second=final_response.generation_tps,
            peak_memory_gb=final_response.peak_memory,
        )
