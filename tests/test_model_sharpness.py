"""Test code for non-batch `model_sharpness()`."""

# pylint: disable=invalid-name

import pytest
import torch
from torch import nn

from src.hesse import model_sharpness

# TODO: Add more test cases


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
        sharpness = model_sharpness(
            model=bilinear,
            inputs=inputs,
            is_batch=False,
        )

    # Check sharpness shape
    output_shape = torch.Size([1])  # Model output is (1)
    assert sharpness.shape == output_shape, "Error in sharpness shape"

    # Check sharpness values
    assert sharpness == 0.0, "Error in sharpness values"
