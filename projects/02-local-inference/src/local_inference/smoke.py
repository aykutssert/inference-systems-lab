import json
import platform
import time
from dataclasses import asdict, dataclass
from typing import Any

from mlx_lm import load, stream_generate

from local_inference.chat import build_prompt
from local_inference.config import MAX_TOKENS, MODEL_ID, MODEL_REVISION, PROMPT


@dataclass(frozen=True)
class SmokeResult:
    model: str
    revision: str
    prompt: str
    response: str
    load_seconds: float
    time_to_first_token_seconds: float
    prompt_tokens: int
    prompt_tokens_per_second: float
    generation_tokens: int
    generation_tokens_per_second: float
    peak_memory_gb: float
    finish_reason: str | None
    python_version: str
    platform: str
    architecture: str


def run_smoke_test() -> SmokeResult:
    load_started_at = time.perf_counter()
    loaded = load(MODEL_ID, revision=MODEL_REVISION)
    model = loaded[0]
    tokenizer = loaded[1]
    load_seconds = time.perf_counter() - load_started_at

    formatted_prompt = build_prompt(tokenizer, PROMPT)
    generation_started_at = time.perf_counter()
    first_token_at: float | None = None
    response_parts: list[str] = []
    final_response: Any = None

    for generation_response in stream_generate(
        model,
        tokenizer,
        formatted_prompt,
        max_tokens=MAX_TOKENS,
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

    return SmokeResult(
        model=MODEL_ID,
        revision=MODEL_REVISION,
        prompt=PROMPT,
        response=response,
        load_seconds=round(load_seconds, 3),
        time_to_first_token_seconds=round(
            first_token_at - generation_started_at,
            3,
        ),
        prompt_tokens=final_response.prompt_tokens,
        prompt_tokens_per_second=round(final_response.prompt_tps, 3),
        generation_tokens=final_response.generation_tokens,
        generation_tokens_per_second=round(final_response.generation_tps, 3),
        peak_memory_gb=round(final_response.peak_memory, 3),
        finish_reason=final_response.finish_reason,
        python_version=platform.python_version(),
        platform=platform.platform(),
        architecture=platform.machine(),
    )


def main() -> None:
    result = run_smoke_test()
    print(result.response)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
