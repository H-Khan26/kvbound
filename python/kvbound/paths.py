"""Repository locations shared by every script."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"
DATA = REPO_ROOT / "data"
FIXTURES = REPO_ROOT / "fixtures"
RESULTS_RAW = REPO_ROOT / "results" / "raw"
REPORTS = REPO_ROOT / "reports"
BUILD = REPO_ROOT / "build"
