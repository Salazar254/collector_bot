"""Latency micro-benchmark harness.

Usage:
    bench = LatencyBenchmark(device="cpu")
    for _ in range(N):
        with bench.stage("ssm_step"):
            output = model(x)
    bench.print_report()
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

import numpy as np
import torch


class LatencyBenchmark:
    """Per-stage latency recorder with CUDA-sync support."""

    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._times: dict[str, list[float]] = {}

    def _sync(self) -> None:
        if self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        """Context manager: time the enclosed block and record in ms."""
        self._sync()
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            self._sync()
            elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            self._times.setdefault(name, []).append(elapsed_ms)

    def report(
        self, percentiles: tuple[int, ...] = (50, 90, 99)
    ) -> dict[str, dict[str, float]]:
        """Return per-stage statistics keyed by stage name."""
        out: dict[str, dict[str, float]] = {}
        for name, times in self._times.items():
            arr = np.asarray(times, dtype=np.float64)
            out[name] = {
                **{f"p{p}": float(np.percentile(arr, p)) for p in percentiles},
                "mean": float(arr.mean()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "n": int(len(arr)),
            }
        return out

    def reset(self) -> None:
        self._times.clear()

    def print_report(self, percentiles: tuple[int, ...] = (50, 90, 99)) -> None:
        rep = self.report(percentiles)
        header = f"{'Stage':<32}" + "".join(f"  p{p:>2}(ms)" for p in percentiles) + "  mean(ms)      n"
        print(header)
        print("-" * len(header))
        for name in sorted(rep):
            s = rep[name]
            row = f"{name:<32}"
            row += "".join(f"  {s[f'p{p}']:>9.3f}" for p in percentiles)
            row += f"  {s['mean']:>9.3f}  {s['n']:>6}"
            print(row)

    def assert_stage_under(self, name: str, max_ms: float, percentile: int = 99) -> None:
        """Raise AssertionError if stage p{percentile} exceeds max_ms."""
        rep = self.report()
        if name not in rep:
            raise KeyError(f"Stage {name!r} not recorded")
        actual = rep[name][f"p{percentile}"]
        assert actual <= max_ms, (
            f"Stage {name!r} p{percentile}={actual:.2f}ms exceeds budget {max_ms}ms"
        )
