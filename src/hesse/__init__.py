"""Main module."""

import importlib.metadata
import os

from src.hesse.hessian_dict import (
    batch_loss_hessian_dict,
    batch_model_hessian_dict,
    loss_hessian_dict,
    model_hessian_dict,
)
from src.hesse.hessian_matrix import (
    batch_loss_hessian_matrix,
    batch_model_hessian_matrix,
    loss_hessian_matrix,
    model_hessian_matrix,
)
from src.hesse.sharpness import (
    batch_loss_sharpness,
    batch_model_sharpness,
    loss_sharpness,
    model_sharpness,
)

__all__ = [
    "batch_loss_hessian_dict",
    "batch_loss_hessian_matrix",
    "batch_loss_sharpness",
    "batch_model_hessian_dict",
    "batch_model_hessian_matrix",
    "batch_model_sharpness",
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
