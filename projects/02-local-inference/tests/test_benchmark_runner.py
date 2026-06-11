from typing import Literal

import pytest

from local_inference.benchmark_runner import GenerationResult, run_benchmark
from local_inference.benchmark_schema import BenchmarkPrompt
from local_inference.config import MEASURED_RUNS, WARMUP_RUNS
from local_inference.prompts import BENCHMARK_PROMPTS


class FakeBackend:
    name: Literal["mlx", "llama.cpp"] = "mlx"
    model = "test-model"
    model_revision = "test-revision"
    runtime_version = "test-runtime"

    def __init__(self) -> None:
        self.load_calls = 0
        self.generated_prompt_ids: list[str] = []

    def load(self) -> float:
        self.load_calls += 1
        return 1.23456

    def generate(self, prompt: BenchmarkPrompt) -> GenerationResult:
        self.generated_prompt_ids.append(prompt.id)
        return GenerationResult(
            response=f"response for {prompt.id}",
            finish_reason="stop",
            time_to_first_token_seconds=0.12345,
            prompt_tokens=10,
            prompt_tokens_per_second=100.12345,
            generation_tokens=5,
            generation_tokens_per_second=80.12345,
            peak_memory_gb=1.12345,
        )


def test_runner_loads_once_and_returns_measured_records() -> None:
    backend = FakeBackend()

    records = run_benchmark(
        backend,
        BENCHMARK_PROMPTS,
        experiment_id="experiment-1",
        recorded_at="2026-06-11T14:00:00+00:00",
    )

    assert backend.load_calls == 1
    assert len(records) == len(BENCHMARK_PROMPTS) * MEASURED_RUNS
    assert backend.generated_prompt_ids == [
        BENCHMARK_PROMPTS[0].id,
        *[prompt.id for prompt in BENCHMARK_PROMPTS for _ in range(MEASURED_RUNS)],
    ]
    assert records[0].run_index == 1
    assert records[MEASURED_RUNS - 1].run_index == MEASURED_RUNS
    assert records[MEASURED_RUNS].run_index == 1
    assert records[0].experiment.warmup_runs == WARMUP_RUNS
    assert records[0].experiment.measured_runs == MEASURED_RUNS
    assert records[0].metrics.load_seconds == 1.235
    assert records[0].metrics.time_to_first_token_seconds == 0.123
    assert records[0].metrics.generation_tokens_per_second == 80.123


def test_runner_rejects_empty_prompt_set() -> None:
    with pytest.raises(ValueError, match="at least one prompt"):
        run_benchmark(
            FakeBackend(),
            (),
            experiment_id="experiment-1",
            recorded_at="2026-06-11T14:00:00+00:00",
        )
