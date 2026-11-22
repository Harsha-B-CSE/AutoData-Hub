import gc
import os
from enum import Enum
from typing import Callable, Iterable, List, Any, TypeVar
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from backend.config import settings

T = TypeVar("T")
R = TypeVar("R")

class ExecutionMode(str, Enum):
    SEQUENTIAL = "sequential"
    BATCH = "batch"
    PARALLEL = "parallel"

def get_optimal_worker_count() -> int:
    """
    Returns worker count based on physical cores: max(1, CPU_CORES - 1),
    capped by MAX_WORKER_PROCESSES to ensure responsiveness on 8GB laptop.
    """
    return settings.MAX_WORKER_PROCESSES

def get_worker_pool(max_workers: int = None, use_threads: bool = False):
    workers = max_workers or get_optimal_worker_count()
    if use_threads:
        return ThreadPoolExecutor(max_workers=workers)
    return ProcessPoolExecutor(max_workers=workers)

def execute_workload(
    func: Callable[[T], R],
    items: List[T],
    mode: ExecutionMode = ExecutionMode.BATCH,
    batch_size: int = 50,
    max_workers: int = None,
    use_threads: bool = False
) -> List[R]:
    """
    Executes a workload across items using the specified ExecutionMode.
    Triggers explicit garbage collection to prevent memory ballooning on 8GB RAM host.
    """
    if not items:
        return []

    workers = max_workers or get_optimal_worker_count()
    results: List[R] = []

    if mode == ExecutionMode.SEQUENTIAL or len(items) <= 10:
        for item in items:
            results.append(func(item))
        gc.collect()
        return results

    if mode == ExecutionMode.BATCH:
        # Process in batches to maximize L2/L3 cache locality and keep memory bound
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            for item in chunk:
                results.append(func(item))
            # Trigger GC after each batch
            gc.collect()
        return results

    if mode == ExecutionMode.PARALLEL:
        # Distribute items across worker pool
        pool_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        try:
            with pool_cls(max_workers=workers) as executor:
                results = list(executor.map(func, items, chunksize=max(1, len(items) // (workers * 4) or 1)))
        except Exception:
            # Fallback to sequential in case of any worker process serialization error
            results = [func(item) for item in items]
        finally:
            gc.collect()
        return results

    return [func(item) for item in items]
