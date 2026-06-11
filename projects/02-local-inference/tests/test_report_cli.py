from pathlib import Path

from local_inference.report_cli import parse_args


def test_parse_args_requires_input_path() -> None:
    args = parse_args(["benchmark.jsonl"])

    assert args.input == Path("benchmark.jsonl")
    assert args.output is None


def test_parse_args_accepts_output_path() -> None:
    args = parse_args(["benchmark.jsonl", "--output", "report.md"])

    assert args.input == Path("benchmark.jsonl")
    assert args.output == Path("report.md")
