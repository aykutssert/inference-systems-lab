from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from local_inference.benchmark_cli import (
    build_experiment_identity,
    default_output_directory,
    parse_args,
    run_cli,
)
from local_inference.benchmark_runner import GenerationResult
from local_inference.benchmark_schema import BenchmarkPrompt
from local_inference.config import MEASURED_RUNS
from local_inference.prompts import BENCHMARK_PROMPTS


class FakeBackend:
    name: Literal["mlx", "llama.cpp"] = "mlx"
    model = "test-model"
    model_revision = "test-revision"
    runtime_version = "test-runtime"

    def load(self) -> float:
        return 1.0

    def generate(self, prompt: BenchmarkPrompt) -> GenerationResult:
        return GenerationResult(
            response=f"response for {prompt.id}",
            finish_reason="stop",
            time_to_first_token_seconds=0.1,
            prompt_tokens=10,
            prompt_tokens_per_second=100.0,
            generation_tokens=5,
            generation_tokens_per_second=80.0,
            peak_memory_gb=1.1,
        )


def test_experiment_identity_uses_utc_timestamp() -> None:
    now = datetime(2026, 6, 11, 14, 30, 15, 123456, tzinfo=UTC)

    experiment_id, recorded_at = build_experiment_identity(now, "mlx")

    assert experiment_id == "mlx-20260611T143015.123456Z"
    assert recorded_at == "2026-06-11T14:30:15.123456+00:00"


def test_experiment_identity_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone"):
        build_experiment_identity(datetime(2026, 6, 11, 14, 30), "mlx")


def test_cli_writes_all_measured_records(tmp_path: Path) -> None:
    output_path = run_cli(
        FakeBackend(),
        tmp_path,
        now=datetime(2026, 6, 11, 14, 30, 15, 123456, tzinfo=UTC),
    )

    assert output_path == tmp_path / "mlx-20260611T143015.123456Z.jsonl"
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == (
        len(BENCHMARK_PROMPTS) * MEASURED_RUNS
    )


def test_cli_output_directory_argument() -> None:
    args = parse_args(["--output-directory", "/tmp/benchmark-output"])

    assert args.output_directory == Path("/tmp/benchmark-output")


def test_default_output_directory_is_inside_repository() -> None:
    assert default_output_directory().parts[-2:] == ("benchmarks", "mlx")
