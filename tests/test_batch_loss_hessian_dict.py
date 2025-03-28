"""Test code for `batch_loss_hessian_dict()`."""

# pylint: disable=invalid-name

from typing import Callable

import pytest
import torch
from torch import nn

from src.hesse import batch_loss_hessian_dict
from src.hesse.types import Criterion

# TODO: Add test cases to exercise `diagonal_only` argument


def test_batch_loss_hessian_dict_bilinear(
    bilinear: nn.Module, commutation_matrix: Callable, mse: Criterion, batch_size: int
) -> None:
    """Test with bilinear model."""
    # Make input data
    m, n = bilinear.B.in1_features, bilinear.B.in2_features
    x1 = torch.randn(batch_size, m).requires_grad_(False)
    x2 = torch.randn(batch_size, n).requires_grad_(False)
    batch_inputs = (x1, x2)
    batch_target = torch.randn(batch_size).requires_grad_(False)

    # PyTorch issues performance warning for unimplemented batching rule
    # - This does not affect the correctness of the implementation.
    with pytest.warns(UserWarning):
        hess = batch_loss_hessian_dict(bilinear, mse, batch_inputs, batch_target)

    # Check Hessian shape
    err_str = "Error in Hessian shape"
    expected_shape = torch.Size([batch_size]) + 2 * bilinear.B.weight.shape
    assert hess["B.weight"]["B.weight"].shape == expected_shape, err_str

    # Check Hessian values
    err_str = "Error in Hessian values"
    K = commutation_matrix(m, n).requires_grad_(False)

    for batch in range(batch_size):
        flat_1 = torch.outer(x1[batch, :], x2[batch, :]).flatten()
        flat_2 = torch.outer(x2[batch, :], x1[batch, :]).flatten()
        expected_value = 2.0 * (flat_1.unsqueeze(-1) @ flat_2.unsqueeze(-1).T @ K.T)
        actual_value = hess["B.weight"]["B.weight"][batch].view(m * n, m * n)
        assert torch.allclose(actual_value, expected_value), err_str
