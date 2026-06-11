from pathlib import Path

import pytest

from local_inference.benchmark_report import (
    escape_cell,
    format_metric,
    generate_report,
    render_markdown,
    write_markdown,
)
from local_inference.benchmark_schema import (
    BenchmarkPrompt,
    BenchmarkRecord,
    ExperimentMetadata,
    RunMetrics,
)
from local_inference.benchmark_summary import (
    BenchmarkSummary,
    MetricSummary,
    PromptSummary,
)


def make_summary() -> BenchmarkSummary:
    return BenchmarkSummary(
        experiment=ExperimentMetadata(
            schema_version="1",
            experiment_id="mlx-20260611T140000.000000Z",
            recorded_at="2026-06-11T14:00:00+00:00",
            backend="mlx",
            model="mlx-community/Qwen3-1.7B-4bit",
            model_revision="revision",
            generation_mode="non-thinking",
            warmup_runs=1,
            measured_runs=3,
            python_version="3.13.13",
            platform="macOS-arm64",
            architecture="arm64",
            runtime_version="0.31.3",
        ),
        load_seconds=1.23456,
        prompts=(
            PromptSummary(
                prompt_id="short-definition",
                category="short",
                run_count=3,
                prompt_tokens=21,
                generation_tokens=26,
                time_to_first_token_seconds=MetricSummary(0.1, 0.2, 0.3),
                prompt_tokens_per_second=MetricSummary(100.0, 110.0, 120.0),
                generation_tokens_per_second=MetricSummary(80.0, 90.0, 100.0),
                peak_memory_gb=MetricSummary(1.0, 1.1, 1.2),
            ),
        ),
    )


def test_format_metric_uses_fixed_precision() -> None:
    assert format_metric(MetricSummary(0.1, 0.25, 1.0)) == ("0.100 / 0.250 / 1.000")


def test_escape_cell_protects_markdown_table() -> None:
    assert escape_cell("value|with\\separator\nnext") == (
        "value\\|with\\\\separator next"
    )


def test_render_markdown_contains_metadata_and_metrics() -> None:
    report = render_markdown(make_summary())

    assert report.startswith("# Benchmark Report: mlx-20260611T140000.000000Z\n")
    assert "| Backend | `mlx` |" in report
    assert "| Model load time | 1.235 s |" in report
    assert "Values use `minimum / average / maximum`." in report
    assert (
        "| `short-definition` | short | 3 | 21 | 26 | "
        "0.100 / 0.200 / 0.300 | "
        "100.000 / 110.000 / 120.000 | "
        "80.000 / 90.000 / 100.000 | "
        "1.000 / 1.100 / 1.200 |"
    ) in report
    assert report.endswith("\n")


def test_render_markdown_is_deterministic() -> None:
    summary = make_summary()

    assert render_markdown(summary) == render_markdown(summary)


def make_record(run_index: int) -> BenchmarkRecord:
    return BenchmarkRecord(
        experiment=make_summary().experiment,
        prompt=BenchmarkPrompt(
            id="short-definition",
            category="short",
            text="Define latency.",
            max_tokens=32,
        ),
        run_index=run_index,
        response="Latency is delay.",
        finish_reason="stop",
        metrics=RunMetrics(
            load_seconds=1.23456,
            time_to_first_token_seconds=0.1 * run_index,
            prompt_tokens=21,
            prompt_tokens_per_second=100.0 + run_index,
            generation_tokens=26,
            generation_tokens_per_second=80.0 + run_index,
            peak_memory_gb=1.0 + (0.1 * run_index),
        ),
    )


def test_write_markdown_publishes_complete_file(tmp_path: Path) -> None:
    output_path = tmp_path / "reports" / "benchmark.md"

    result = write_markdown("# Report\n", output_path)

    assert result == output_path
    assert output_path.read_text(encoding="utf-8") == "# Report\n"
    assert [
        path for path in output_path.parent.iterdir() if path.suffix == ".tmp"
    ] == []


def test_write_markdown_refuses_to_overwrite(tmp_path: Path) -> None:
    output_path = tmp_path / "benchmark.md"
    output_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_markdown("replacement\n", output_path)

    assert output_path.read_text(encoding="utf-8") == "existing\n"
    assert [path for path in tmp_path.iterdir() if path.suffix == ".tmp"] == []


def test_write_markdown_rejects_empty_report(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty report"):
        write_markdown("", tmp_path / "benchmark.md")


def test_generate_report_uses_input_basename(tmp_path: Path) -> None:
    input_path = tmp_path / "benchmark.jsonl"
    input_path.write_text(
        "".join(f"{make_record(run_index).to_json()}\n" for run_index in range(1, 4)),
        encoding="utf-8",
    )

    output_path = generate_report(input_path)

    assert output_path == tmp_path / "benchmark.md"
    assert output_path.read_text(encoding="utf-8").startswith(
        "# Benchmark Report: mlx-20260611T140000.000000Z\n"
    )


def test_generate_report_rejects_non_jsonl_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\.jsonl"):
        generate_report(tmp_path / "benchmark.json")
