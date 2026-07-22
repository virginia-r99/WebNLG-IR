#!/usr/bin/env python3
"""Lightweight profiling utilities for WebNLG IR experiment runs."""
from __future__ import annotations

import json
import os
import platform
import threading
import time
from pathlib import Path
from typing import Any

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None


class ResourceProfiler:
    """
    Lightweight runtime and memory profiler.

    It records:
      - wall-clock times for named stages
      - total wall-clock time
      - peak process RSS memory sampled in a background thread
      - peak CUDA allocated/reserved memory, when available

    The profiler is intentionally dependency-light. If psutil is unavailable,
    CPU RSS is omitted. If torch/CUDA is unavailable, GPU fields are omitted.
    """

    def __init__(self, enabled: bool = True, sample_interval: float = 0.25, device: str | None = None):
        self.enabled = bool(enabled)
        self.sample_interval = float(sample_interval)
        self.device = device
        self.stage_times: dict[str, float] = {}
        self.stage_starts: dict[str, float] = {}
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.peak_rss_bytes = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.process = psutil.Process(os.getpid()) if psutil is not None else None

    def start(self) -> None:
        if not self.enabled:
            return
        self.start_time = time.perf_counter()
        self._reset_cuda_peak_memory()
        self._sample_once()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self.end_time = time.perf_counter()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample_once()

    def start_stage(self, name: str) -> None:
        if not self.enabled:
            return
        self.stage_starts[str(name)] = time.perf_counter()

    def stop_stage(self, name: str) -> None:
        if not self.enabled:
            return
        name = str(name)
        start = self.stage_starts.pop(name, None)
        if start is None:
            return
        elapsed = time.perf_counter() - start
        self.stage_times[name] = self.stage_times.get(name, 0.0) + elapsed
        self._sample_once()

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            self._sample_once()
            time.sleep(self.sample_interval)

    def _sample_once(self) -> None:
        if self.process is None:
            return
        try:
            rss = int(self.process.memory_info().rss)
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        except Exception:
            pass

    def _reset_cuda_peak_memory(self) -> None:
        if torch is None or not torch.cuda.is_available():
            return
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def cuda_stats(self) -> dict[str, Any]:
        if torch is None or not torch.cuda.is_available():
            return {
                "cuda_available": False,
                "gpu_name": None,
                "peak_gpu_allocated_mb": None,
                "peak_gpu_reserved_mb": None,
            }
        try:
            return {
                "cuda_available": True,
                "gpu_name": torch.cuda.get_device_name(0),
                "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
                "peak_gpu_reserved_mb": torch.cuda.max_memory_reserved() / (1024 ** 2),
            }
        except Exception:
            return {
                "cuda_available": True,
                "gpu_name": None,
                "peak_gpu_allocated_mb": None,
                "peak_gpu_reserved_mb": None,
            }

    def summary(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        total_time_s = None
        if self.start_time is not None:
            end = self.end_time if self.end_time is not None else time.perf_counter()
            total_time_s = end - self.start_time

        data: dict[str, Any] = {
            "total_time_s": total_time_s,
            "peak_cpu_rss_mb": (self.peak_rss_bytes / (1024 ** 2)) if self.peak_rss_bytes else None,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        }
        for k, v in sorted(self.stage_times.items()):
            data[f"{k}_time_s"] = v
        data.update(self.cuda_stats())
        if extra:
            data.update(extra)
        return data

    def write_json(self, path: str | Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.summary(extra=extra)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved profile to {path}")
        return data


def profile_path_from_output(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    return output_path.with_suffix(output_path.suffix + ".profile.json")


def safe_rate(n: int | float | None, seconds: int | float | None) -> float | None:
    if n is None or seconds is None:
        return None
    try:
        seconds = float(seconds)
        if seconds <= 0:
            return None
        return float(n) / seconds
    except Exception:
        return None


def safe_ms_per_item(seconds: int | float | None, n: int | float | None) -> float | None:
    if n is None or seconds is None:
        return None
    try:
        n = float(n)
        seconds = float(seconds)
        if n <= 0:
            return None
        return (seconds / n) * 1000.0
    except Exception:
        return None
