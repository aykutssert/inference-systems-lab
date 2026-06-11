import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path

from local_inference.benchmark_schema import BenchmarkRecord

EXPERIMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def write_jsonl(
    records: Sequence[BenchmarkRecord],
    output_directory: Path,
) -> Path:
    if not records:
        raise ValueError("Cannot write an empty benchmark")

    experiment_id = records[0].experiment.experiment_id
    if not EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ValueError("Experiment id is not safe for a file name")
    if any(record.experiment.experiment_id != experiment_id for record in records):
        raise ValueError("All records must belong to the same experiment")

    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{experiment_id}.jsonl"
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_directory,
            prefix=f".{experiment_id}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            for record in records:
                temporary_file.write(record.to_json())
                temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.link(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return output_path
