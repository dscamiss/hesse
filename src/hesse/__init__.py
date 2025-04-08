"""Main module."""

import importlib.metadata
import os

from src.hesse.hessian_dict import loss_hessian_dict, model_hessian_dict
from src.hesse.hessian_matrix import loss_hessian_matrix, model_hessian_matrix
from src.hesse.sharpness import loss_sharpness, model_sharpness

__all__ = [
    "loss_hessian_dict",
    "loss_hessian_matrix",
    "loss_sharpness",
    "model_hessian_dict",
    "model_hessian_matrix",
    "model_sharpness",
]

# The call to `importlib.metadata.version()` fails in GitHub CI
if not os.getenv("GITHUB_ACTIONS"):
    __version__ = importlib.metadata.version("hesse")
