import json

import pytest

from local_inference.benchmark_schema import (
    BenchmarkPrompt,
    BenchmarkRecord,
    ExperimentMetadata,
    RunMetrics,
)
from local_inference.prompts import BENCHMARK_PROMPTS


def test_benchmark_prompt_ids_are_unique() -> None:
    prompt_ids = [prompt.id for prompt in BENCHMARK_PROMPTS]

    assert len(prompt_ids) == len(set(prompt_ids))
    assert {prompt.category for prompt in BENCHMARK_PROMPTS} == {
        "short",
        "medium",
        "long-context",
    }
    long_context_prompt = next(
        prompt for prompt in BENCHMARK_PROMPTS if prompt.category == "long-context"
    )
    assert len(long_context_prompt.text) > 5_000


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", ""),
        ("text", ""),
        ("max_tokens", 0),
    ],
)
def test_benchmark_prompt_rejects_invalid_values(
    field: str,
    value: str | int,
) -> None:
    values: dict[str, str | int] = {
        "id": "valid",
        "category": "short",
        "text": "valid",
        "max_tokens": 1,
    }
    values[field] = value

    with pytest.raises(ValueError):
        BenchmarkPrompt(**values)  # type: ignore[arg-type]


def test_benchmark_record_serializes_as_single_json_line() -> None:
    record = BenchmarkRecord(
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
        prompt=BENCHMARK_PROMPTS[0],
        run_index=1,
        response="Latency is response delay.",
        finish_reason="stop",
        metrics=RunMetrics(
            load_seconds=1.0,
            time_to_first_token_seconds=0.2,
            prompt_tokens=10,
            prompt_tokens_per_second=100.0,
            generation_tokens=5,
            generation_tokens_per_second=80.0,
            peak_memory_gb=1.1,
        ),
    )

    serialized = record.to_json()
    payload = json.loads(serialized)

    assert "\n" not in serialized
    assert payload["experiment"]["backend"] == "mlx"
    assert payload["prompt"]["id"] == "short-definition"
    assert payload["metrics"]["generation_tokens_per_second"] == 80.0


@pytest.mark.parametrize(
    ("warmup_runs", "measured_runs"),
    [
        (-1, 3),
        (1, 0),
    ],
)
def test_experiment_metadata_rejects_invalid_run_counts(
    warmup_runs: int,
    measured_runs: int,
) -> None:
    with pytest.raises(ValueError):
        ExperimentMetadata(
            schema_version="1",
            experiment_id="experiment-1",
            recorded_at="2026-06-11T14:00:00+00:00",
            backend="mlx",
            model="model",
            model_revision="revision",
            generation_mode="non-thinking",
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            python_version="3.13.13",
            platform="macOS",
            architecture="arm64",
            runtime_version="0.31.3",
        )


def test_run_metrics_reject_negative_values() -> None:
    with pytest.raises(ValueError):
        RunMetrics(
            load_seconds=1.0,
            time_to_first_token_seconds=-0.1,
            prompt_tokens=10,
            prompt_tokens_per_second=100.0,
            generation_tokens=5,
            generation_tokens_per_second=80.0,
            peak_memory_gb=1.1,
        )


@pytest.mark.parametrize(
    ("run_index", "response"),
    [
        (0, "valid"),
        (1, ""),
    ],
)
def test_benchmark_record_rejects_invalid_values(
    run_index: int,
    response: str,
) -> None:
    with pytest.raises(ValueError):
        BenchmarkRecord(
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
            prompt=BENCHMARK_PROMPTS[0],
            run_index=run_index,
            response=response,
            finish_reason="stop",
            metrics=RunMetrics(
                load_seconds=1.0,
                time_to_first_token_seconds=0.2,
                prompt_tokens=10,
                prompt_tokens_per_second=100.0,
                generation_tokens=5,
                generation_tokens_per_second=80.0,
                peak_memory_gb=1.1,
            ),
        )
