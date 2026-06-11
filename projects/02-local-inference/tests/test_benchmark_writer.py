import json
from dataclasses import replace
from pathlib import Path

import pytest

from local_inference.benchmark_schema import (
    BenchmarkPrompt,
    BenchmarkRecord,
    ExperimentMetadata,
    RunMetrics,
)
from local_inference.benchmark_writer import write_jsonl


def make_record(
    *,
    experiment_id: str = "experiment-1",
    run_index: int = 1,
) -> BenchmarkRecord:
    return BenchmarkRecord(
        experiment=ExperimentMetadata(
            schema_version="1",
            experiment_id=experiment_id,
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
            id="short",
            category="short",
            text="Define latency.",
            max_tokens=32,
        ),
        run_index=run_index,
        response="Latency is delay.",
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


def test_writer_creates_one_json_line_per_record(tmp_path: Path) -> None:
    records = [make_record(run_index=1), make_record(run_index=2)]

    output_path = write_jsonl(records, tmp_path / "nested")
    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert output_path == tmp_path / "nested" / "experiment-1.jsonl"
    assert len(lines) == 2
    assert [json.loads(line)["run_index"] for line in lines] == [1, 2]
    assert list(output_path.parent.glob("*.tmp")) == []


def test_writer_does_not_overwrite_existing_result(tmp_path: Path) -> None:
    output_path = tmp_path / "experiment-1.jsonl"
    output_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_jsonl([make_record()], tmp_path)

    assert output_path.read_text(encoding="utf-8") == "existing\n"
    assert [path for path in tmp_path.iterdir() if path.suffix == ".tmp"] == []


def test_writer_rejects_empty_records(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty benchmark"):
        write_jsonl([], tmp_path)


def test_writer_rejects_mixed_experiments(tmp_path: Path) -> None:
    first = make_record()
    second = replace(
        first,
        experiment=replace(first.experiment, experiment_id="experiment-2"),
    )

    with pytest.raises(ValueError, match="same experiment"):
        write_jsonl([first, second], tmp_path)


@pytest.mark.parametrize("experiment_id", ["../escape", "/absolute", "space id"])
def test_writer_rejects_unsafe_experiment_id(
    tmp_path: Path,
    experiment_id: str,
) -> None:
    with pytest.raises(ValueError, match="safe for a file name"):
        write_jsonl([make_record(experiment_id=experiment_id)], tmp_path)
