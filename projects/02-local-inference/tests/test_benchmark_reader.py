import json
from pathlib import Path

import pytest

from local_inference.benchmark_reader import BenchmarkReadError, read_jsonl
from local_inference.benchmark_schema import (
    BenchmarkPrompt,
    BenchmarkRecord,
    ExperimentMetadata,
    RunMetrics,
)


def make_record(*, run_index: int = 1) -> BenchmarkRecord:
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


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_reader_loads_valid_records(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.jsonl"
    write_lines(
        path,
        [
            make_record(run_index=1).to_json(),
            make_record(run_index=2).to_json(),
        ],
    )

    records = read_jsonl(path)

    assert [record.run_index for record in records] == [1, 2]
    assert records[0].experiment.experiment_id == "experiment-1"


@pytest.mark.parametrize(
    ("lines", "line_number", "message"),
    [
        (["{"], 1, "Expecting property name"),
        ([json.dumps(["not", "an", "object"])], 1, "JSON object"),
        ([""], 1, "Blank line"),
        ([json.dumps({"experiment": {}})], 1, "Missing record fields"),
        (
            [
                json.dumps(
                    {
                        **json.loads(make_record().to_json()),
                        "unexpected": True,
                    }
                )
            ],
            1,
            "Unexpected record fields",
        ),
        (
            [
                json.dumps(
                    {
                        **json.loads(make_record().to_json()),
                        "metrics": [],
                    }
                )
            ],
            1,
            "metrics",
        ),
    ],
)
def test_reader_reports_malformed_line(
    tmp_path: Path,
    lines: list[str],
    line_number: int,
    message: str,
) -> None:
    path = tmp_path / "benchmark.jsonl"
    write_lines(path, lines)

    with pytest.raises(BenchmarkReadError) as error_info:
        read_jsonl(path)

    assert error_info.value.path == path
    assert error_info.value.line_number == line_number
    assert message in str(error_info.value)


def test_reader_reports_later_line_number(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.jsonl"
    write_lines(path, [make_record().to_json(), "invalid"])

    with pytest.raises(BenchmarkReadError) as error_info:
        read_jsonl(path)

    assert error_info.value.line_number == 2
    assert f"{path}:2:" in str(error_info.value)


def test_reader_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.jsonl"
    path.touch()

    with pytest.raises(BenchmarkReadError, match="empty"):
        read_jsonl(path)
