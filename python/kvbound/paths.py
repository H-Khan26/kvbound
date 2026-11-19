"""Repository locations shared by every script."""

import os
import platform
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"
DATA = REPO_ROOT / "data"
FIXTURES = REPO_ROOT / "fixtures"
RESULTS_RAW = REPO_ROOT / "results" / "raw"
REPORTS = REPO_ROOT / "reports"
BUILD = REPO_ROOT / "build"
# Outside stacks' binaries, converted weights and helper environments; $KVBOUND_EXT moves them
# off the repository's drive.
EXT = Path(os.environ.get("KVBOUND_EXT") or REPO_ROOT / "ext")


def binary(name: str, preset: str) -> Path:
    """build/<preset>/bin/kvb-<name>; exits with a hint when it is not built."""
    suffix = ".exe" if platform.system() == "Windows" else ""
    path = BUILD / preset / "bin" / f"kvb-{name}{suffix}"
    if not path.exists():
        raise SystemExit(f"{path} not found; run `make build PRESET={preset}` first")
    return path
