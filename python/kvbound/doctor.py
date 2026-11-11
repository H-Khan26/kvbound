"""Checks that this machine can produce trustworthy benchmark numbers.

Usage: python -m kvbound.doctor   (exit code 1 if any check fails)
"""

import shutil
import sys
import time
from pathlib import Path

import pynvml
import torch

from kvbound import fingerprint as fp

OK, WARN, FAIL = "ok", "warn", "FAIL"
Check = tuple[str, str, str]  # (status, what, detail)


def _sm_clock_idle_range(seconds: float = 1.0) -> tuple[int, int, int]:
    """(min, max, ceiling) SM clock in MHz over a short idle window."""
    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        ceiling = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM)
        seen = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            seen.append(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
            time.sleep(0.05)
        return min(seen), max(seen), ceiling
    finally:
        pynvml.nvmlShutdown()


def _perf_counter_check() -> Check:
    """Nsight Compute needs GPU performance counters readable by non-root users."""
    params = Path("/proc/driver/nvidia/params")
    if not params.exists():
        return WARN, "perf counters", "cannot check on this OS (run Nsight Compute elevated)"
    admin_only = "RmProfilingAdminOnly: 1" in params.read_text()
    if admin_only:
        return FAIL, "perf counters", "admin only; set NVreg_RestrictProfilingToAdminUsers=0"
    return OK, "perf counters", "readable by non-root users"


def checks() -> list[Check]:
    info = fp.fingerprint()
    gpu, sw = info["gpu"], info["software"]
    out: list[Check] = [
        (OK, "gpu", f"{gpu['name']} ({info['device_id']}), driver {gpu['driver_version']}")
    ]

    if info["device_config"]:
        out.append((OK, "device config", f"configs/devices/{info['device_config']}.toml"))
    else:
        out.append((FAIL, "device config", f"no configs/devices/*.toml named {gpu['name']!r}"))

    if sw["os_class"] == "linux":
        out.append((OK, "os", sw["os"]))
    else:
        detail = f"{sw['os']}: clocks cannot be locked; records are tagged os={sw['os_class']}"
        out.append((WARN, "os", detail))

    if torch.cuda.is_available():
        out.append((OK, "torch", f"{sw['torch']} with CUDA {sw['torch_cuda']}"))
    else:
        out.append((FAIL, "torch", f"{sw['torch']} cannot see the GPU (install the cu128 extra)"))

    out.append((OK, "nvcc", sw["nvcc"]) if sw["nvcc"] else (FAIL, "nvcc", "not on PATH"))
    for tool, name in (("nsys", "Nsight Systems"), ("ncu", "Nsight Compute")):
        found = shutil.which(tool)
        out.append((OK, tool, found) if found else (WARN, tool, f"{name} not on PATH"))
    out.append(_perf_counter_check())

    lo, hi, ceiling = _sm_clock_idle_range()
    if lo == hi and sw["os_class"] == "linux":
        out.append((OK, "clocks", f"SM clock steady at {lo} MHz (ceiling {ceiling})"))
    else:
        fix = "run scripts/lock_clocks.sh" if sw["os_class"] == "linux" else "needs native Linux"
        out.append((WARN, "clocks", f"SM clock idles at {lo}-{hi} MHz, not locked ({fix})"))

    if gpu["persistence_mode"] is False:
        out.append((WARN, "persistence", "off; run scripts/lock_clocks.sh"))
    elif gpu["persistence_mode"]:
        out.append((OK, "persistence", "on"))

    if info["git"]["dirty"]:
        out.append((WARN, "git", "tree is dirty; benchmarks need --allow-dirty"))
    else:
        out.append((OK, "git", info["git"]["sha"][:12]))
    return out


def main() -> None:
    results = checks()
    for status, what, detail in results:
        print(f"[{status:>4}] {what:<14} {detail}")
    statuses = [status for status, _, _ in results]
    print(f"\n{statuses.count(FAIL)} failed, {statuses.count(WARN)} warnings")
    sys.exit(1 if FAIL in statuses else 0)


if __name__ == "__main__":
    main()
