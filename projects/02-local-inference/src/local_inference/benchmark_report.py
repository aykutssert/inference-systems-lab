import os
import tempfile
from pathlib import Path

from local_inference.benchmark_reader import read_jsonl
from local_inference.benchmark_summary import (
    BenchmarkSummary,
    MetricSummary,
    summarize_benchmark,
)


def escape_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def format_metric(metric: MetricSummary) -> str:
    return f"{metric.minimum:.3f} / {metric.average:.3f} / {metric.maximum:.3f}"


def render_markdown(summary: BenchmarkSummary) -> str:
    experiment = summary.experiment
    lines = [
        f"# Benchmark Report: {experiment.experiment_id}",
        "",
        "## Experiment",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Backend | `{escape_cell(experiment.backend)}` |",
        f"| Model | `{escape_cell(experiment.model)}` |",
        f"| Model revision | `{escape_cell(experiment.model_revision)}` |",
        f"| Runtime version | `{escape_cell(experiment.runtime_version)}` |",
        f"| Recorded at | `{escape_cell(experiment.recorded_at)}` |",
        f"| Platform | `{escape_cell(experiment.platform)}` |",
        f"| Python | `{escape_cell(experiment.python_version)}` |",
        f"| Warm-up runs | {experiment.warmup_runs} |",
        f"| Measured runs per prompt | {experiment.measured_runs} |",
        f"| Model load time | {summary.load_seconds:.3f} s |",
        "",
        "## Prompt Metrics",
        "",
        "Values use `minimum / average / maximum`.",
        "",
        (
            "| Prompt | Category | Runs | Prompt tokens | Generation tokens | "
            "TTFT (s) | Prompt tokens/s | Generation tokens/s | Peak memory (GB) |"
        ),
        ("| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |"),
    ]

    for prompt in summary.prompts:
        lines.append(
            "| "
            f"`{escape_cell(prompt.prompt_id)}` | "
            f"{escape_cell(prompt.category)} | "
            f"{prompt.run_count} | "
            f"{prompt.prompt_tokens} | "
            f"{prompt.generation_tokens} | "
            f"{format_metric(prompt.time_to_first_token_seconds)} | "
            f"{format_metric(prompt.prompt_tokens_per_second)} | "
            f"{format_metric(prompt.generation_tokens_per_second)} | "
            f"{format_metric(prompt.peak_memory_gb)} |"
        )

    return "\n".join(lines) + "\n"


def write_markdown(report: str, output_path: Path) -> Path:
    if not report:
        raise ValueError("Cannot write an empty report")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(report)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.link(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return output_path


def generate_report(input_path: Path, output_path: Path | None = None) -> Path:
    if input_path.suffix != ".jsonl":
        raise ValueError("Benchmark input must use the .jsonl extension")

    resolved_output_path = output_path or input_path.with_suffix(".md")
    records = read_jsonl(input_path)
    summary = summarize_benchmark(records)
    report = render_markdown(summary)
    return write_markdown(report, resolved_output_path)
