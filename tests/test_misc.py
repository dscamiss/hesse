"""Catch-all for miscellaneous test code."""

import importlib
import os
import sys
from types import ModuleType
from unittest import mock

import hesse


@mock.patch.dict(os.environ, {"GITHUB_ACTIONS": "true"})
def test_github_actions_env_effect():
    """Test `GITHUB_ACTIONS` environment variable effect."""

    def reload_package() -> ModuleType:
        """Reload the package to reflect environment variable changes."""
        importlib.invalidate_caches()
        return importlib.import_module("hesse")  # noqa: DCO030

    # Ensure a fresh import
    if "hesse" in sys.modules:
        del sys.modules["hesse"]
    _hesse = reload_package()

    assert not hasattr(_hesse, "__version__"), "Unexpected __version__ attribute"


def test_no_github_actions_env_effect():
    """Test no `GITHUB_ACTIONS` environment variable effect."""
    assert hesse.__version__, "Missing __version__ attribute"
