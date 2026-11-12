"""Shared fixtures for the GPU tests."""

import pytest

from kvbound import native


@pytest.fixture(scope="session")
def kvb():
    """The _kvb extension module; skips the test when it is not built."""
    try:
        return native.load()
    except ImportError as e:
        pytest.skip(str(e))
