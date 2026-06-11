import json
from pathlib import Path
from typing import Any

from local_inference.benchmark_schema import (
    BenchmarkPrompt,
    BenchmarkRecord,
    ExperimentMetadata,
    RunMetrics,
)

RECORD_FIELDS = {
    "experiment",
    "prompt",
    "run_index",
    "response",
    "finish_reason",
    "metrics",
}


class BenchmarkReadError(ValueError):
    def __init__(self, path: Path, line_number: int, message: str) -> None:
        super().__init__(f"{path}:{line_number}: {message}")
        self.path = path
        self.line_number = line_number


def parse_record(payload: Any) -> BenchmarkRecord:
    if not isinstance(payload, dict):
        raise TypeError("Record must be a JSON object")
    payload_fields = set(payload)
    missing_fields = sorted(RECORD_FIELDS - payload_fields)
    extra_fields = sorted(payload_fields - RECORD_FIELDS)
    if missing_fields:
        raise ValueError(f"Missing record fields: {', '.join(missing_fields)}")
    if extra_fields:
        raise ValueError(f"Unexpected record fields: {', '.join(extra_fields)}")
    for field in ("experiment", "prompt", "metrics"):
        if not isinstance(payload[field], dict):
            raise TypeError(f"Record field '{field}' must be a JSON object")

    return BenchmarkRecord(
        experiment=ExperimentMetadata(**payload["experiment"]),
        prompt=BenchmarkPrompt(**payload["prompt"]),
        run_index=payload["run_index"],
        response=payload["response"],
        finish_reason=payload["finish_reason"],
        metrics=RunMetrics(**payload["metrics"]),
    )


def read_jsonl(path: Path) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []

    with path.open(encoding="utf-8") as result_file:
        for line_number, line in enumerate(result_file, start=1):
            if not line.strip():
                raise BenchmarkReadError(path, line_number, "Blank line is not allowed")
            try:
                payload = json.loads(line)
                records.append(parse_record(payload))
            except (
                AttributeError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                raise BenchmarkReadError(path, line_number, str(error)) from error

    if not records:
        raise BenchmarkReadError(path, 1, "Benchmark file is empty")

    return records
