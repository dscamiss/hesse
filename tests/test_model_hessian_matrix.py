"""Test code for `model_hessian_matrix()`."""

# pylint: disable=invalid-name

from typing import Callable

import pytest
import torch
from torch import nn

from src.hesse import model_hessian_matrix


def test_model_hessian_matrix_bilinear(bilinear: nn.Module) -> None:
    """Test with bilinear model."""
    # Make input data
    x1 = torch.randn(bilinear.B.in1_features).requires_grad_(False)
    x2 = torch.randn(bilinear.B.in2_features).requires_grad_(False)
    inputs = (x1, x2)

    # PyTorch issues performance warning for unimplemented batching rule
    # - This does not affect the correctness of the implementation.
    with pytest.warns(UserWarning):
        hess = model_hessian_matrix(bilinear, inputs)

    # Check Hessian shape
    err_str = "Error in Hessian shape"
    expected_shape = 2 * torch.Size([bilinear.B.weight.numel()])
    assert hess.shape == expected_shape, err_str

    # Check Hessian values
    err_str = "Error in Hessian values"
    assert torch.all(hess == 0.0), err_str


@pytest.mark.parametrize("diagonal_only", [True, False])
def test_model_hessian_dict_double_bilinear(
    double_bilinear: nn.Module, commutation_matrix: Callable, diagonal_only: bool
) -> None:
    """Test with double-bilinear model."""
    # Make aliases for brevity
    B1 = double_bilinear.B1
    B2 = double_bilinear.B2
    m, n, p = B1.shape[0], B1.shape[1], B2.shape[1]

    # Make input data
    x1 = torch.randn(m).requires_grad_(False)
    x2 = torch.randn(p).requires_grad_(False)
    inputs = (x1, x2)

    # Compute Hessian matrix
    hessian_matrix = model_hessian_matrix(double_bilinear, inputs, diagonal_only=diagonal_only)

    # Check Hessian matrix shape
    err_str = "Error in Hessian matrix shape"
    expected_shape = 2 * torch.Size([(m * n) + (n * p)])
    assert hessian_matrix.shape == expected_shape, err_str

    # Check Hessian matrix values
    # - See comments in `test_model_hessian_dict.py` for derivations
    err_str = "Error in Hessian matrix values"
    K = commutation_matrix(m, n).requires_grad_(False)

    # Check (B1, B1) Hessian matrix values
    assert torch.all(hessian_matrix[: (m * n), : (m * n)] == 0.0), err_str

    # Check (B1, B2) Hessian matrix values
    if not diagonal_only:
        expected_value = K @ torch.kron(torch.eye(n), torch.outer(x1, x2))
    else:
        expected_value = 0.0
    assert torch.all(hessian_matrix[: (m * n), (m * n) :] == expected_value), err_str

    # Check (B2, B1) Hessian matrix values
    if not diagonal_only:
        expected_value = torch.kron(torch.eye(n), torch.outer(x2, x1)) @ K.T
    else:
        expected_value = 0.0
    assert torch.all(hessian_matrix[(m * n) :, : (m * n)] == expected_value), err_str

    # Check (B2, B2) Hessian matrix values
    assert torch.all(hessian_matrix[(m * n) :, (m * n) :] == 0.0), err_str
