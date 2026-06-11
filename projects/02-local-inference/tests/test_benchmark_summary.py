from dataclasses import replace

import pytest

from local_inference.benchmark_schema import (
    BenchmarkPrompt,
    BenchmarkRecord,
    ExperimentMetadata,
    RunMetrics,
)
from local_inference.benchmark_summary import summarize_benchmark


def make_record(
    *,
    prompt_id: str = "short",
    category: str = "short",
    run_index: int = 1,
    ttft: float = 0.1,
    generation_tps: float = 80.0,
    peak_memory_gb: float = 1.0,
) -> BenchmarkRecord:
    return BenchmarkRecord(
        experiment=ExperimentMetadata(
            schema_version="1",
            experiment_id="experiment-1",
            recorded_at="2026-06-11T14:00:00+00:00",
            backend="mlx",
            model="model",
            model_revision="revision",
            generation_mode="non-thinking",
            warmup_runs=1,
            measured_runs=3,
            python_version="3.13.13",
            platform="macOS",
            architecture="arm64",
            runtime_version="0.31.3",
        ),
        prompt=BenchmarkPrompt(
            id=prompt_id,
            category=category,  # type: ignore[arg-type]
            text="Prompt text",
            max_tokens=64,
        ),
        run_index=run_index,
        response="Response text",
        finish_reason="stop",
        metrics=RunMetrics(
            load_seconds=1.25,
            time_to_first_token_seconds=ttft,
            prompt_tokens=20,
            prompt_tokens_per_second=100.0 + run_index,
            generation_tokens=10,
            generation_tokens_per_second=generation_tps,
            peak_memory_gb=peak_memory_gb,
        ),
    )


def test_summary_aggregates_each_prompt() -> None:
    records = [
        make_record(run_index=1, ttft=0.1, generation_tps=80.0),
        make_record(run_index=2, ttft=0.2, generation_tps=90.0),
        make_record(run_index=3, ttft=0.3, generation_tps=100.0),
        make_record(
            prompt_id="medium",
            category="medium",
            run_index=1,
            ttft=0.4,
            generation_tps=70.0,
            peak_memory_gb=1.2,
        ),
        make_record(
            prompt_id="medium",
            category="medium",
            run_index=2,
            ttft=0.5,
            generation_tps=75.0,
            peak_memory_gb=1.2,
        ),
        make_record(
            prompt_id="medium",
            category="medium",
            run_index=3,
            ttft=0.6,
            generation_tps=80.0,
            peak_memory_gb=1.2,
        ),
    ]

    summary = summarize_benchmark(records)

    assert summary.experiment.experiment_id == "experiment-1"
    assert summary.load_seconds == 1.25
    assert [prompt.prompt_id for prompt in summary.prompts] == ["medium", "short"]
    short = summary.prompts[1]
    assert short.run_count == 3
    assert short.prompt_tokens == 20
    assert short.generation_tokens == 10
    assert short.time_to_first_token_seconds.minimum == 0.1
    assert short.time_to_first_token_seconds.average == 0.2
    assert short.time_to_first_token_seconds.maximum == 0.3
    assert short.generation_tokens_per_second.average == 90.0
    assert short.peak_memory_gb.average == 1.0


def test_summary_rejects_empty_records() -> None:
    with pytest.raises(ValueError, match="empty benchmark"):
        summarize_benchmark([])


def test_summary_rejects_mixed_experiments() -> None:
    first = make_record()
    second = replace(
        first,
        experiment=replace(first.experiment, experiment_id="experiment-2"),
    )

    with pytest.raises(ValueError, match="same experiment"):
        summarize_benchmark([first, second])


def test_summary_rejects_missing_prompt_runs() -> None:
    with pytest.raises(ValueError, match="run count"):
        summarize_benchmark([make_record()])


def test_summary_rejects_mixed_prompt_definitions() -> None:
    first = make_record(run_index=1)
    second = replace(
        make_record(run_index=2),
        prompt=replace(first.prompt, text="Different prompt text"),
    )
    third = make_record(run_index=3)

    with pytest.raises(ValueError, match="Prompt definition"):
        summarize_benchmark([first, second, third])


@pytest.mark.parametrize(
    "metrics",
    [
        RunMetrics(
            load_seconds=2.0,
            time_to_first_token_seconds=0.1,
            prompt_tokens=20,
            prompt_tokens_per_second=100.0,
            generation_tokens=10,
            generation_tokens_per_second=80.0,
            peak_memory_gb=1.0,
        ),
        RunMetrics(
            load_seconds=1.25,
            time_to_first_token_seconds=0.1,
            prompt_tokens=21,
            prompt_tokens_per_second=100.0,
            generation_tokens=10,
            generation_tokens_per_second=80.0,
            peak_memory_gb=1.0,
        ),
        RunMetrics(
            load_seconds=1.25,
            time_to_first_token_seconds=0.1,
            prompt_tokens=20,
            prompt_tokens_per_second=100.0,
            generation_tokens=11,
            generation_tokens_per_second=80.0,
            peak_memory_gb=1.0,
        ),
    ],
)
def test_summary_rejects_unstable_experiment_values(metrics: RunMetrics) -> None:
    first = make_record()
    second = replace(first, run_index=2, metrics=metrics)
    third = replace(first, run_index=3)

    with pytest.raises(ValueError):
        summarize_benchmark([first, second, third])
