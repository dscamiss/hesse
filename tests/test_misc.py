"""Catch-all for miscellaneous test code."""

import hesse
import importlib
import os
import sys
from types import ModuleType
from unittest import mock

_PACKAGE_NAME = hesse.__name__


def fresh_import() -> ModuleType:
    """
    Reload the package to reflect environment variable changes.

    Returns:
        Reloaded package.
    """
    if _PACKAGE_NAME in sys.modules:
        del sys.modules[_PACKAGE_NAME]

    importlib.invalidate_caches()
    return importlib.import_module(_PACKAGE_NAME)


@mock.patch.dict(os.environ, {"GITHUB_ACTIONS": "true"})
def test_github_actions_env_effect() -> None:
    """Test `GITHUB_ACTIONS` environment variable effect."""
    _hesse = fresh_import()
    assert not hasattr(_hesse, "__version__"), "Unexpected __version__ attribute"


@mock.patch.dict(os.environ, clear=True)
def test_no_github_actions_env_effect() -> None:
    """Test no `GITHUB_ACTIONS` environment variable effect."""
    _hesse = fresh_import()
    assert _hesse.__version__, "Missing __version__ attribute"
