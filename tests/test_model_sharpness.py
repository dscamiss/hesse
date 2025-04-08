"""Test code for non-batch `model_sharpness()`."""

# pylint: disable=invalid-name

import pytest
import torch
from torch import nn

from src.hesse import model_sharpness


def test_model_sharpness_bilinear(bilinear: nn.Module) -> None:
    """Non-batch `model_sharpness()` with bilinear model."""
    # Make inputs
    x1 = torch.randn(bilinear.B.in1_features).requires_grad_(False)
    x2 = torch.randn(bilinear.B.in2_features).requires_grad_(False)
    inputs = (x1, x2)

    # Compute sharpness
    # - PyTorch issues performance warning for unimplemented batching rule
    # - This does not affect the correctness of the implementation.
    with pytest.warns(UserWarning):
        sharpness = model_sharpness(model=bilinear, inputs=inputs)

    assert sharpness == 0.0, "Error in sharpness value"


def test_model_sharpness_double_bilinear(double_bilinear: nn.Module) -> None:
    """Non-batch `model_sharpness()` with double-bilinear model."""
    # Make aliases for brevity
    B1 = double_bilinear.B1
    B2 = double_bilinear.B2
    m, p = B1.shape[0], B2.shape[1]

    # Make inputs
    x1 = torch.randn(m).requires_grad_(False)
    x2 = torch.randn(p).requires_grad_(False)
    inputs = (x1, x2)

    # Compute sharpness
    model_sharpness(model=double_bilinear, inputs=inputs)

    # TODO
