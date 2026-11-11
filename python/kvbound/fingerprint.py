"""Describes the machine a run came from, so result files from different boxes can be compared.

Usage: python -m kvbound.fingerprint
"""

import json
import platform
import re
import shutil
import subprocess
import tomllib
from typing import Any

import pynvml
import torch

from kvbound.paths import CONFIGS, REPO_ROOT


def _nvml_or_none(query: Any, *args: Any) -> Any:
    """Runs an NVML query that some platforms or driver models do not support."""
    try:
        return query(*args)
    except pynvml.NVMLError:
        return None


def gpu_info(index: int = 0) -> dict[str, Any]:
    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(index)
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
        power_mw = _nvml_or_none(pynvml.nvmlDeviceGetPowerManagementLimit, h)
        persistence = _nvml_or_none(pynvml.nvmlDeviceGetPersistenceMode, h)
        return {
            "name": pynvml.nvmlDeviceGetName(h),
            "uuid": pynvml.nvmlDeviceGetUUID(h),
            "compute_capability": f"{major}.{minor}",
            "memory_total_mib": pynvml.nvmlDeviceGetMemoryInfo(h).total >> 20,
            "driver_version": pynvml.nvmlSystemGetDriverVersion(),
            "sm_clock_max_mhz": _nvml_or_none(
                pynvml.nvmlDeviceGetMaxClockInfo, h, pynvml.NVML_CLOCK_SM
            ),
            "pcie_gen_max": _nvml_or_none(pynvml.nvmlDeviceGetMaxPcieLinkGeneration, h),
            "pcie_width_max": _nvml_or_none(pynvml.nvmlDeviceGetMaxPcieLinkWidth, h),
            "power_limit_w": None if power_mw is None else power_mw / 1000,
            "persistence_mode": None if persistence is None else bool(persistence),
        }
    finally:
        pynvml.nvmlShutdown()


def device_id(gpu: dict[str, Any]) -> str:
    """Short stable id for a physical card, e.g. rtx3060ti-fd7b40d5."""
    model = re.sub(r"[^a-z0-9]", "", gpu["name"].lower().replace("nvidia geforce", ""))
    return f"{model}-{gpu['uuid'].removeprefix('GPU-')[:8]}"


def device_config(gpu_name: str) -> tuple[str, dict[str, Any]] | None:
    """The configs/devices/*.toml entry whose name matches this GPU, as (stem, table)."""
    for path in sorted((CONFIGS / "devices").glob("*.toml")):
        table = tomllib.loads(path.read_text(encoding="utf-8"))
        if table["name"] == gpu_name:
            return path.stem, table
    return None


def os_class() -> str:
    """linux | wsl | windows | other. Only native linux can lock clocks and run every profiler."""
    system = platform.system()
    if system == "Linux":
        return "wsl" if "microsoft" in platform.release().lower() else "linux"
    return "windows" if system == "Windows" else "other"


def _tool_version(tool: str, pattern: str) -> str | None:
    exe = shutil.which(tool)
    if exe is None:
        return None
    out = subprocess.run([exe, "--version"], capture_output=True, text=True, check=False).stdout
    match = re.search(pattern, out)
    return match.group(1) if match else "unknown"


def software_info() -> dict[str, Any]:
    return {
        "os_class": os_class(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nvcc": _tool_version("nvcc", r"release ([\d.]+)"),
        "cmake": _tool_version("cmake", r"version ([\d.]+)"),
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def git_state() -> dict[str, Any]:
    """Commit and whether the tree differs from it. Result and report outputs do not count."""
    changes = _git("status", "--porcelain", "--", ".", ":(exclude)results", ":(exclude)reports")
    return {"sha": _git("rev-parse", "HEAD"), "dirty": bool(changes)}


def fingerprint(gpu_index: int = 0) -> dict[str, Any]:
    gpu = gpu_info(gpu_index)
    config = device_config(gpu["name"])
    return {
        "device_id": device_id(gpu),
        "device_config": config[0] if config else None,
        "gpu": gpu,
        "software": software_info(),
        "git": git_state(),
    }


if __name__ == "__main__":
    print(json.dumps(fingerprint(), indent=2))
