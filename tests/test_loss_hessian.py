"""Test code for `loss_hessian()`."""

# pylint: disable=invalid-name

from typing import Callable

import pytest
import torch
from torch import nn

from src.hesse import loss_hessian
from src.hesse.types import Criterion


def test_loss_hessian_bilinear(
    bilinear: nn.Module, commutation_matrix: Callable, mse: Criterion
) -> None:
    """Test with bilinear model."""
    # Make input data
    m, n = bilinear.B.in1_features, bilinear.B.in2_features
    x1 = torch.randn(m).requires_grad_(False)
    x2 = torch.randn(n).requires_grad_(False)
    inputs = (x1, x2)
    target = torch.randn([]).requires_grad_(False)

    # PyTorch issues performance warning for unimplemented batching rule
    # - This does not affect the correctness of the implementation.
    with pytest.warns(UserWarning):
        hess = loss_hessian(bilinear, mse, inputs, target)

    # Check Hessian shape
    err_str = "Error in Hessian shape"
    expected_shape = 2 * bilinear.B.weight.shape
    assert hess["B.weight"]["B.weight"].shape == expected_shape, err_str

    # Check Hessian values
    # - Note that
    #
    #       d^2 f(B) . (V, W)
    #           = 2 <x^t V y, x^t W y>
    #           = 2 tr(x^t V y) tr(x^t W y)
    #           = 2 tr(V y x^t) tr(y x^t W)
    #           = 2 vec(V^t)^t vec(y x^t) vec(x y^t)^t vec(W)
    #           = 2 flat(V) vec(y x^t) vec(x y^t)^t K^t flat(W)
    #           = 2 flat(V) flat(x y^t) flat(y x^t)^t K^t flat(W),
    #
    #   where
    #
    #   * tr() is the trace function,
    #   * vec() is the column-stacking vectorization map,
    #   * flat() is the row-stacking vectorization map,
    #   * K is the commutation matrix K_{m,n}.
    #
    # - This means that the Hessian is
    #      Hess = 2 flat(x y^t) flat(y x^t)^t K^t.

    err_str = "Error in Hessian values"
    K = commutation_matrix(m, n).requires_grad_(False)

    flat_1 = torch.outer(x1, x2).flatten()
    flat_2 = torch.outer(x2, x1).flatten()
    expected_value = 2.0 * (flat_1.unsqueeze(-1) @ flat_2.unsqueeze(-1).T @ K.T)
    assert torch.allclose(hess["B.weight"]["B.weight"].view(m * n, m * n), expected_value), err_str
