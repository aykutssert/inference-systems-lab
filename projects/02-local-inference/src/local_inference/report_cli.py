import argparse
from collections.abc import Sequence
from pathlib import Path

from local_inference.benchmark_report import generate_report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown report from benchmark JSONL results."
    )
    parser.add_argument("input", type=Path, help="Benchmark JSONL file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output Markdown path. Defaults to the input path with a .md suffix.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_path = generate_report(args.input, args.output)
    print(output_path)
