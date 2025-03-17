"""Test code for utility functions."""

# pylint: disable=invalid-name

from typing import Callable

import torch
from torch import nn

from src.hesse import model_hessian
from src.hesse import make_hessian_matrix


@torch.no_grad()
def test_commutation_matrix(commutation_matrix: Callable) -> None:
    """Test `commutation_matrix()`."""
    A = torch.randn(3, 4)
    K = commutation_matrix(A.shape[0], A.shape[1])

    err_str = "Error in commutation matrix"
    # Note: Transpose here since `flatten()` uses row-major component ordering
    vec_A = A.T.flatten()
    vec_A_transpose = A.flatten()
    assert torch.all(K @ vec_A == vec_A_transpose), err_str


@torch.no_grad()
def test_make_hessian_matrix(double_bilinear: nn.Module) -> None:
    """Test `make_hessian_matrix()` with double-bilinear model."""
    # Make aliases for brevity
    B1 = double_bilinear.B1
    B2 = double_bilinear.B2
    m, n, p = B1.shape[0], B1.shape[1], B2.shape[1]

    # Make input data
    x1 = torch.randn(m).requires_grad_(False)
    x2 = torch.randn(p).requires_grad_(False)
    inputs = (x1, x2)

    # Compute Hessian
    hessian = model_hessian(double_bilinear, inputs)

    # Make Hessian matrix
    hessian_matrix = make_hessian_matrix(double_bilinear, hessian)

    # Check matrix entries
    err_str = "Error in Hessian matrix values"
    A = hessian["B1"]["B1"].view(m * n, m * n)
    B = hessian["B1"]["B2"].view(m * n, n * p)
    C = hessian["B2"]["B1"].view(n * p, m * n)
    D = hessian["B2"]["B2"].view(n * p, n * p)
    row_0 = torch.cat((A, B), dim=1)
    row_1 = torch.cat((C, D), dim=1)
    expected_hessian_matrix = torch.cat((row_0, row_1), dim=0)
    assert torch.all(hessian_matrix == expected_hessian_matrix), err_str
