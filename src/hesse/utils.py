"""Utility functions."""

from typing import Any


def make_tuple(obj: Any) -> tuple:
    """
    Make object into tuple.

    Args:
        obj: Input object.

    Returns:
        This function returns `obj` if `obj` is a tuple.  Otherwise, it
        returns the 1-tuple containing `obj`.
    """
    return obj if isinstance(obj, tuple) else (obj,)
