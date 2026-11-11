"""Samples GPU clock, temperature, power and memory in the background while a benchmark runs.

Asking the driver to lock clocks does not guarantee they held (thermal or power limits can still
throttle), so every record carries what the clock actually was.
"""

import threading
import time
from types import TracebackType
from typing import Any

import pynvml

METRICS = ("sm_clock_mhz", "mem_clock_mhz", "temp_c", "power_w", "util_pct", "mem_used_mib")


class NvmlSampler:
    """Context manager: samples every `interval_s` between enter and exit."""

    def __init__(self, gpu_index: int = 0, interval_s: float = 0.1) -> None:
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.t: list[float] = []
        self.samples: dict[str, list[float]] = {name: [] for name in METRICS}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self, handle: Any) -> None:
        self.t.append(time.perf_counter())
        values = (
            pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
            pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM),
            pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
            pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
            pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            pynvml.nvmlDeviceGetMemoryInfo(handle).used >> 20,
        )
        for name, value in zip(METRICS, values, strict=True):
            self.samples[name].append(value)

    def _run(self) -> None:
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            while True:
                self._sample(handle)
                if self._stop.wait(self.interval_s):
                    break
            self._sample(handle)
        finally:
            pynvml.nvmlShutdown()

    def __enter__(self) -> "NvmlSampler":
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        self._thread.join()

    def summary(self) -> dict[str, Any]:
        """min / mean / max of each metric over the run, plus the sampling settings."""
        out: dict[str, Any] = {"interval_s": self.interval_s, "n_samples": len(self.t)}
        for name, values in self.samples.items():
            if values:
                out[name] = {
                    "min": min(values),
                    "mean": sum(values) / len(values),
                    "max": max(values),
                }
        return out
