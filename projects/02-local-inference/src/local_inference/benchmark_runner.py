import platform
from dataclasses import dataclass
from typing import Literal, Protocol

from local_inference.benchmark_schema import (
    BenchmarkPrompt,
    BenchmarkRecord,
    ExperimentMetadata,
    RunMetrics,
)
from local_inference.config import MEASURED_RUNS, WARMUP_RUNS


@dataclass(frozen=True)
class GenerationResult:
    response: str
    finish_reason: str | None
    time_to_first_token_seconds: float
    prompt_tokens: int
    prompt_tokens_per_second: float
    generation_tokens: int
    generation_tokens_per_second: float
    peak_memory_gb: float


class BenchmarkBackend(Protocol):
    name: Literal["mlx", "llama.cpp"]
    model: str
    model_revision: str
    runtime_version: str

    def load(self) -> float: ...

    def generate(self, prompt: BenchmarkPrompt) -> GenerationResult: ...


def run_benchmark(
    backend: BenchmarkBackend,
    prompts: tuple[BenchmarkPrompt, ...],
    *,
    experiment_id: str,
    recorded_at: str,
) -> list[BenchmarkRecord]:
    if not prompts:
        raise ValueError("Benchmark requires at least one prompt")

    load_seconds = backend.load()

    for _ in range(WARMUP_RUNS):
        backend.generate(prompts[0])

    experiment = ExperimentMetadata(
        schema_version="1",
        experiment_id=experiment_id,
        recorded_at=recorded_at,
        backend=backend.name,
        model=backend.model,
        model_revision=backend.model_revision,
        generation_mode="non-thinking",
        warmup_runs=WARMUP_RUNS,
        measured_runs=MEASURED_RUNS,
        python_version=platform.python_version(),
        platform=platform.platform(),
        architecture=platform.machine(),
        runtime_version=backend.runtime_version,
    )

    records: list[BenchmarkRecord] = []
    for prompt in prompts:
        for run_index in range(1, MEASURED_RUNS + 1):
            result = backend.generate(prompt)
            records.append(
                BenchmarkRecord(
                    experiment=experiment,
                    prompt=prompt,
                    run_index=run_index,
                    response=result.response,
                    finish_reason=result.finish_reason,
                    metrics=RunMetrics(
                        load_seconds=round(load_seconds, 3),
                        time_to_first_token_seconds=round(
                            result.time_to_first_token_seconds,
                            3,
                        ),
                        prompt_tokens=result.prompt_tokens,
                        prompt_tokens_per_second=round(
                            result.prompt_tokens_per_second,
                            3,
                        ),
                        generation_tokens=result.generation_tokens,
                        generation_tokens_per_second=round(
                            result.generation_tokens_per_second,
                            3,
                        ),
                        peak_memory_gb=round(result.peak_memory_gb, 3),
                    ),
                )
            )
    return records
