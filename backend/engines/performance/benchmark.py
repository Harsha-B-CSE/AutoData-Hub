import time
import os
import psutil
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

@dataclass
class StageBenchmark:
    stage_name: str
    latency_ms: float
    memory_delta_mb: float
    cpu_percent: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "latency_ms": round(self.latency_ms, 2),
            "memory_delta_mb": round(self.memory_delta_mb, 2),
            "cpu_percent": round(self.cpu_percent, 2),
            "extra": self.extra,
        }

class BenchmarkContext:
    """
    Context manager to accurately measure latency (ms), RSS memory delta (MB),
    and CPU utilization for an execution stage.
    """
    def __init__(self, stage_name: str):
        self.stage_name = stage_name
        self.process = psutil.Process(os.getpid())
        self.start_time: float = 0.0
        self.start_mem: float = 0.0
        self.result: Optional[StageBenchmark] = None

    def __enter__(self):
        # Warm up cpu_percent measurement
        self.process.cpu_percent(interval=None)
        self.start_mem = self.process.memory_info().rss / (1024 * 1024)
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        end_mem = self.process.memory_info().rss / (1024 * 1024)
        cpu = self.process.cpu_percent(interval=None)

        latency_ms = (end_time - self.start_time) * 1000.0
        memory_delta_mb = end_mem - self.start_mem

        self.result = StageBenchmark(
            stage_name=self.stage_name,
            latency_ms=max(0.01, latency_ms),
            memory_delta_mb=memory_delta_mb,
            cpu_percent=cpu,
        )
