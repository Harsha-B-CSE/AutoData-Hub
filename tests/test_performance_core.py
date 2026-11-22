import time
import pytest
from backend.engines.performance.benchmark import BenchmarkContext
from backend.engines.performance.cache import PipelineCache
from backend.engines.performance.pool_manager import ExecutionMode, execute_workload

def test_benchmark_context():
    with BenchmarkContext("TestStage") as b:
        # Simulate slight work
        time.sleep(0.02)

    assert b.result is not None
    assert b.result.stage_name == "TestStage"
    assert b.result.latency_ms >= 15.0
    assert isinstance(b.result.memory_delta_mb, float)
    assert isinstance(b.result.cpu_percent, float)

def test_pipeline_cache():
    key1 = PipelineCache.generate_cache_key("sha256abc", "tabular", {"scaling": "minmax"})
    key2 = PipelineCache.generate_cache_key("sha256abc", "tabular", {"scaling": "minmax"})
    key3 = PipelineCache.generate_cache_key("sha256xyz", "tabular", {"scaling": "minmax"})

    assert key1 == key2
    assert key1 != key3

    PipelineCache.set(key1, {"output_path": "/dummy/path.csv", "pre_score": 50, "post_score": 90})
    cached = PipelineCache.get(key1)
    assert cached is not None
    assert cached["pre_score"] == 50

def test_execution_modes():
    items = list(range(100))
    def double(x):
        return x * 2

    # Sequential
    res_seq = execute_workload(double, items, mode=ExecutionMode.SEQUENTIAL)
    assert res_seq == [x * 2 for x in items]

    # Batch
    res_batch = execute_workload(double, items, mode=ExecutionMode.BATCH, batch_size=25)
    assert res_batch == [x * 2 for x in items]

    # Parallel
    res_par = execute_workload(double, items, mode=ExecutionMode.PARALLEL, use_threads=True)
    assert res_par == [x * 2 for x in items]
