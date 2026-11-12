"""Loads _kvb, the C++/CUDA engine built by CMake as a Python extension.

The extension lives in build/<preset>/python rather than in the package, so a source checkout
without a GPU build still imports kvbound. Build it with `make build PRESET=cuda`.
"""

import importlib
import os
import sys
from types import ModuleType

from kvbound.paths import BUILD


def load(preset: str = "cuda") -> ModuleType:
    """Returns the _kvb module from build/<preset>/python; ImportError if it is not built."""
    directory = BUILD / preset / "python"
    if not any(directory.glob("_kvb.*")):
        raise ImportError(f"_kvb is not built in {directory}; run `make build PRESET={preset}`")
    if sys.platform == "win32" and "CUDA_PATH" in os.environ:
        # Extension modules do not search PATH for the DLLs they depend on (cuBLAS).
        os.add_dll_directory(os.path.join(os.environ["CUDA_PATH"], "bin"))
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    return importlib.import_module("_kvb")
