from collections.abc import Iterator, Sequence
from typing import Any

from local_inference.chat import ChatMessage, build_prompt
from local_inference.mlx_backend import MlxBackend
from mlx_lm import stream_generate

from production_serving.streaming import GenerationChunk


class StreamingMlxBackend(MlxBackend):
    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> Iterator[GenerationChunk]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Backend must be loaded before generation")

        formatted_prompt = build_prompt(self._tokenizer, messages)
        final_response: Any = None
        for generation_response in stream_generate(
            self._model,
            self._tokenizer,
            formatted_prompt,
            max_tokens=max_tokens,
        ):
            yield GenerationChunk(text=generation_response.text)
            final_response = generation_response

        if final_response is None:
            raise RuntimeError("Model produced no generation response")

        yield GenerationChunk(
            text="",
            finish_reason=final_response.finish_reason,
            prompt_tokens=final_response.prompt_tokens,
            generation_tokens=final_response.generation_tokens,
        )
