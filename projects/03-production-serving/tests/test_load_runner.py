import pytest

from production_serving.load_runner import RequestResult, build_report, percentile


def test_percentile_uses_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    assert percentile(values, 0.50) == 2.5
    assert percentile(values, 0.95) == pytest.approx(3.85)
    assert percentile([], 0.95) is None


def test_build_report_summarizes_success_and_errors() -> None:
    results = [
        RequestResult(0, 200, 0.2, 1.0, 10, None),
        RequestResult(1, 200, 0.4, 2.0, 20, None),
        RequestResult(2, 429, None, 0.1, 0, "server_busy"),
    ]

    report = build_report(results, concurrency=2, wall_time_seconds=2.0)
    summary = report["summary"]

    assert isinstance(summary, dict)
    assert summary["requests"] == 3
    assert summary["successful"] == 2
    assert summary["errors"] == 1
    assert summary["error_rate"] == pytest.approx(1 / 3)
    assert summary["status_counts"] == {"200": 2, "429": 1}
    assert summary["requests_per_second"] == 1.5
    assert summary["completion_tokens_per_second"] == 15.0
    assert summary["latency_seconds"] == {
        "mean": 1.5,
        "p50": 1.5,
        "p95": 1.95,
        "p99": 1.99,
    }
