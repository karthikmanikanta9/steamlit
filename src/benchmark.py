"""Consistent timing and memory measurement (Phase 5).

Time: `repeats` runs, each timed with perf_counter, tracemalloc OFF.
Memory: one separate run with tracemalloc ON; the reported number is the peak
of Python/NumPy memory allocated DURING the run (the algorithm's own working
structures). The input Graph arrays already existed and are not counted.
"""
import gc
import statistics
import time
import tracemalloc
from dataclasses import dataclass, field

from src import config
from src.mst import MSTResult
from src.mst_validation import all_passed, validate_mst


@dataclass
class BenchmarkResult:
    algorithm: str
    result: MSTResult
    runtimes: list
    peak_memory_bytes: int
    checks: list = field(default_factory=list)

    @property
    def repeats(self) -> int:
        return len(self.runtimes)

    @property
    def median_seconds(self) -> float:
        return statistics.median(self.runtimes)

    @property
    def min_seconds(self) -> float:
        return min(self.runtimes)

    @property
    def valid(self) -> bool:
        return bool(self.checks) and all_passed(self.checks)


def benchmark_algorithm(name, func, graph, repeats=config.TIMING_REPEATS, **kwargs):
    runtimes, result = [], None
    for _ in range(max(1, repeats)):
        gc.collect()
        started = time.perf_counter()
        result = func(graph, **kwargs)
        runtimes.append(time.perf_counter() - started)

    gc.collect()
    tracemalloc.start()
    try:
        func(graph, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return BenchmarkResult(name, result, runtimes, int(peak))


def run_and_validate(name, func, graph, repeats, reference_weight=None, **kwargs):
    bench = benchmark_algorithm(name, func, graph, repeats, **kwargs)
    bench.checks = validate_mst(graph, bench.result, reference_weight)
    return bench