import argparse
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from local_inference.benchmark_runner import BenchmarkBackend, run_benchmark
from local_inference.benchmark_writer import write_jsonl
from local_inference.prompts import BENCHMARK_PROMPTS


def default_output_directory() -> Path:
    search_roots = (Path.cwd(), Path(__file__).resolve())
    for search_root in search_roots:
        for candidate in (search_root, *search_root.parents):
            if (candidate / "Roadmap.md").is_file() and (
                candidate / "projects"
            ).is_dir():
                return candidate / "benchmarks" / "mlx"
    raise RuntimeError("Repository root not found. Use --output-directory explicitly.")


def build_experiment_identity(now: datetime, backend_name: str) -> tuple[str, str]:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Experiment timestamp must include a timezone")
    recorded_at = now.astimezone(UTC).isoformat()
    experiment_id = f"{backend_name}-{now.astimezone(UTC):%Y%m%dT%H%M%S.%fZ}"
    return experiment_id, recorded_at


def run_cli(
    backend: BenchmarkBackend,
    output_directory: Path,
    *,
    now: datetime,
) -> Path:
    experiment_id, recorded_at = build_experiment_identity(now, backend.name)
    records = run_benchmark(
        backend,
        BENCHMARK_PROMPTS,
        experiment_id=experiment_id,
        recorded_at=recorded_at,
    )
    return write_jsonl(records, output_directory)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pinned MLX benchmark and write JSONL results.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="Result directory. Defaults to the repository benchmarks/mlx path.",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> None:
    args = parse_args(argv)

    from local_inference.mlx_backend import MlxBackend

    output_path = run_cli(
        MlxBackend(),
        args.output_directory or default_output_directory(),
        now=now(),
    )
    print(output_path)
