"""Test code for batch `model_sharpness()`."""

# pylint: disable=invalid-name

import pytest
import torch
from torch import nn

from src.hesse import model_sharpness

# TODO: Add more test cases


def test_batch_model_sharpness_bilinear(bilinear: nn.Module, batch_size: int) -> None:
    """Batch `model_sharpness()` with bilinear model."""
    # Make inputs
    x1 = torch.randn((batch_size, bilinear.B.in1_features)).requires_grad_(False)
    x2 = torch.randn((batch_size, bilinear.B.in2_features)).requires_grad_(False)
    batch_inputs = (x1, x2)

    # Compute batch sharpness
    # - PyTorch issues performance warning for unimplemented batching rule
    # - This does not affect the correctness of the implementation.
    with pytest.warns(UserWarning):
        batch_sharpness = model_sharpness(
            model=bilinear,
            inputs=batch_inputs,
        )

    # Check batch sharpness shape
    output_shape = torch.Size([batch_size, 1])  # Model output is (batch_size, 1)
    assert batch_sharpness.shape == output_shape, "Error in batch sharpness shape"

    # Check sharpness values
    assert torch.all(batch_sharpness == 0.0), "Error in sharpness values"
