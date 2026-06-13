import pytest

from nvidia_inference.benchmark import (
    RequestResult,
    build_report,
    calculate_tpot,
)


def test_calculate_tpot_excludes_first_token_latency() -> None:
    assert calculate_tpot([1.0, 1.1, 1.2], 3) == pytest.approx(0.1)
    assert calculate_tpot([1.0], 1) is None


def test_build_report_includes_tpot_and_throughput() -> None:
    results = [
        RequestResult(0, 200, 10, 20, 0.2, 0.01, 1.0, None),
        RequestResult(1, 200, 10, 20, 0.4, 0.02, 2.0, None),
        RequestResult(2, 500, 0, 0, None, None, 0.1, "backend_error"),
    ]

    report = build_report(
        results,
        model="test-model",
        concurrency=2,
        wall_time_seconds=2.0,
    )
    summary = report["summary"]

    assert isinstance(summary, dict)
    assert summary["successful"] == 2
    assert summary["errors"] == 1
    assert summary["output_tokens_per_second"] == 20.0
    assert summary["time_per_output_token_seconds"] == {
        "mean": 0.015,
        "p50": 0.015,
        "p95": pytest.approx(0.0195),
        "p99": pytest.approx(0.0199),
    }
