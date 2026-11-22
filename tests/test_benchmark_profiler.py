"""CI-sized checks for the scalability profiler contract."""

from utils.benchmark_profiler import profile_pipeline, to_markdown


def test_scalability_profiler_returns_required_metrics():
    report = profile_pipeline(sizes=(100, 250), random_state=11)
    assert list(report.columns) == [
        "Dataset Size (N)",
        "Wall-Clock Latency (sec)",
        "Peak RAM Footprint (MB)",
        "Throughput (Rows/sec)",
    ]
    assert report["Dataset Size (N)"].tolist() == [100, 250]
    assert (report["Wall-Clock Latency (sec)"] > 0).all()
    assert (report["Peak RAM Footprint (MB)"] > 0).all()
    assert (report["Throughput (Rows/sec)"] > 0).all()
    markdown = to_markdown(report)
    assert "Dataset Size (N)" in markdown
    assert "Throughput (Rows/sec)" in markdown
