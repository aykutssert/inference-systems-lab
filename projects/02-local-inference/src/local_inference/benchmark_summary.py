from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import fmean

from local_inference.benchmark_schema import BenchmarkRecord, ExperimentMetadata


@dataclass(frozen=True)
class MetricSummary:
    minimum: float
    average: float
    maximum: float


@dataclass(frozen=True)
class PromptSummary:
    prompt_id: str
    category: str
    run_count: int
    prompt_tokens: int
    generation_tokens: int
    time_to_first_token_seconds: MetricSummary
    prompt_tokens_per_second: MetricSummary
    generation_tokens_per_second: MetricSummary
    peak_memory_gb: MetricSummary


@dataclass(frozen=True)
class BenchmarkSummary:
    experiment: ExperimentMetadata
    load_seconds: float
    prompts: tuple[PromptSummary, ...]


def summarize_metric(
    records: Sequence[BenchmarkRecord],
    value: Callable[[BenchmarkRecord], float],
) -> MetricSummary:
    values = [value(record) for record in records]
    return MetricSummary(
        minimum=round(min(values), 3),
        average=round(fmean(values), 3),
        maximum=round(max(values), 3),
    )


def summarize_benchmark(records: Sequence[BenchmarkRecord]) -> BenchmarkSummary:
    if not records:
        raise ValueError("Cannot summarize an empty benchmark")

    experiment = records[0].experiment
    if any(record.experiment != experiment for record in records):
        raise ValueError("All records must belong to the same experiment")

    grouped_records: dict[str, list[BenchmarkRecord]] = defaultdict(list)
    for record in records:
        grouped_records[record.prompt.id].append(record)

    prompt_summaries: list[PromptSummary] = []
    for prompt_id in sorted(grouped_records):
        prompt_records = grouped_records[prompt_id]
        first_record = prompt_records[0]
        if len(prompt_records) != experiment.measured_runs:
            raise ValueError("Prompt run count does not match experiment metadata")
        if any(record.prompt != first_record.prompt for record in prompt_records):
            raise ValueError("Prompt definition must be stable within each group")
        prompt_token_counts = {
            record.metrics.prompt_tokens for record in prompt_records
        }
        generation_token_counts = {
            record.metrics.generation_tokens for record in prompt_records
        }
        if len(prompt_token_counts) != 1 or len(generation_token_counts) != 1:
            raise ValueError("Token counts must be stable within each prompt")

        prompt_summaries.append(
            PromptSummary(
                prompt_id=prompt_id,
                category=first_record.prompt.category,
                run_count=len(prompt_records),
                prompt_tokens=first_record.metrics.prompt_tokens,
                generation_tokens=first_record.metrics.generation_tokens,
                time_to_first_token_seconds=summarize_metric(
                    prompt_records,
                    lambda record: record.metrics.time_to_first_token_seconds,
                ),
                prompt_tokens_per_second=summarize_metric(
                    prompt_records,
                    lambda record: record.metrics.prompt_tokens_per_second,
                ),
                generation_tokens_per_second=summarize_metric(
                    prompt_records,
                    lambda record: record.metrics.generation_tokens_per_second,
                ),
                peak_memory_gb=summarize_metric(
                    prompt_records,
                    lambda record: record.metrics.peak_memory_gb,
                ),
            )
        )

    load_seconds = records[0].metrics.load_seconds
    if any(record.metrics.load_seconds != load_seconds for record in records):
        raise ValueError("Load time must be stable within an experiment")

    return BenchmarkSummary(
        experiment=experiment,
        load_seconds=load_seconds,
        prompts=tuple(prompt_summaries),
    )
