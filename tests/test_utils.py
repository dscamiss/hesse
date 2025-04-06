"""Test code for utility functions."""

# pylint: disable=invalid-name

import pytest
import torch
from torch import nn

from src.hesse import model_hessian_dict
from src.hesse.hessian_dict import select_hessian_params
from src.hesse.hessian_matrix import (
    batch_hessian_matrix_from_hessian_dict,
    hessian_matrix_from_hessian_dict,
)
from tests.conftest import commutation_matrix


def test_select_hessian_params() -> None:
    """Test `select_hessian_params()`."""
    model = nn.Linear(2, 3)
    model_param_dict = dict(model.named_parameters())

    # Select both parameters
    param_dict = select_hessian_params(model)

    # Check keys and values
    assert sorted(list(param_dict.keys())) == ["bias", "weight"], "Key error"
    assert torch.all(param_dict["bias"] == model_param_dict["bias"]), "Value error"
    assert torch.all(param_dict["weight"] == model_param_dict["weight"]), "Value error"

    # Select single parameter
    param_dict = select_hessian_params(model, ["bias"])

    # Check keys and values
    assert list(param_dict.keys()) == ["bias"], "Key error"
    assert torch.all(param_dict["bias"] == model_param_dict["bias"]), "Value error"

    # Select nonexistent parameters
    with pytest.raises(ValueError):
        select_hessian_params(model, ["bad_1", "bad_2"])

    # Select no parameters
    with pytest.raises(ValueError):
        select_hessian_params(model, [])


@torch.no_grad()
def test_commutation_matrix() -> None:
    """Test `commutation_matrix()`."""
    A = torch.randn(3, 4)
    K = commutation_matrix(A.shape[0], A.shape[1])

    err_str = "Error in commutation matrix"
    # Note: Transpose here since `flatten()` uses row-major component ordering
    vec_A = A.T.flatten()
    vec_A_transpose = A.flatten()
    assert torch.all(K @ vec_A == vec_A_transpose), err_str


@torch.no_grad()
@pytest.mark.parametrize("diagonal_only", [True, False])
def test_hessian_matrix_from_hessian_dict(double_bilinear: nn.Module, diagonal_only: bool) -> None:
    """Test `hessian_matrix_from_hessian_dict()` with double-bilinear model."""
    # Make aliases for brevity
    B1 = double_bilinear.B1
    B2 = double_bilinear.B2
    m, n, p = B1.shape[0], B1.shape[1], B2.shape[1]

    # Make input data
    x1 = torch.randn(m).requires_grad_(False)
    x2 = torch.randn(p).requires_grad_(False)
    inputs = (x1, x2)

    # Compute Hessian dict
    hessian_dict = model_hessian_dict(
        model=double_bilinear,
        inputs=inputs,
        diagonal_only=diagonal_only,
    )

    # Make Hessian matrix
    hessian_matrix = hessian_matrix_from_hessian_dict(
        model=double_bilinear,
        hessian_dict=hessian_dict,
        diagonal_only=diagonal_only,
    )

    # Check Hessian matrix shape
    err_str = "Error in Hessian matrix shape"
    expected_shape = 2 * torch.Size([(m * n) + (n * p)])
    assert hessian_matrix.shape == expected_shape, err_str

    # Check Hessian matrix values
    err_str = "Error in Hessian matrix values"

    # Compute Hessian matrix blocks
    A = hessian_dict["B1"]["B1"].view(m * n, m * n)
    if not diagonal_only:
        B = hessian_dict["B1"]["B2"].view(m * n, n * p)
        C = hessian_dict["B2"]["B1"].view(n * p, m * n)
    else:
        B = torch.zeros(m * n, n * p)
        C = torch.zeros(n * p, m * n)
    D = hessian_dict["B2"]["B2"].view(n * p, n * p)

    # Assemble expected Hessian matrix from blocks
    row_0 = torch.cat((A, B), dim=1)
    row_1 = torch.cat((C, D), dim=1)
    expected_hessian_matrix = torch.cat((row_0, row_1), dim=0)

    assert torch.all(hessian_matrix == expected_hessian_matrix), err_str


@torch.no_grad()
@pytest.mark.parametrize("diagonal_only", [True, False])
def test_batch_hessian_matrix_from_hessian_dict(
    double_bilinear: nn.Module, diagonal_only: bool, batch_size: int
) -> None:
    """Test `batch_hessian_matrix_from_hessian_dict()` with double-bilinear model."""
    # Make aliases for brevity
    B1 = double_bilinear.B1
    B2 = double_bilinear.B2
    m, n, p = B1.shape[0], B1.shape[1], B2.shape[1]

    # Make input data
    x1 = torch.randn(batch_size, m).requires_grad_(False)
    x2 = torch.randn(batch_size, p).requires_grad_(False)
    batch_inputs = (x1, x2)

    # Compute Hessian dict
    batch_hessian_dict = model_hessian_dict(
        model=double_bilinear,
        inputs=batch_inputs,
        diagonal_only=diagonal_only,
    )

    # Make Hessian matrix
    batch_hessian_matrix = batch_hessian_matrix_from_hessian_dict(
        model=double_bilinear,
        batch_hessian_dict=batch_hessian_dict,
        diagonal_only=diagonal_only,
    )

    # Check Hessian matrix shape
    err_str = "Error in Hessian matrix shape"
    expected_shape = torch.Size([batch_size]) + 2 * torch.Size([(m * n) + (n * p)])
    assert batch_hessian_matrix.shape == expected_shape, err_str

    # Check Hessian matrix values
    err_str = "Error in Hessian matrix values"

    for batch in range(batch_size):
        # Compute Hessian matrix blocks
        A = batch_hessian_dict["B1"]["B1"][batch, :].view(m * n, m * n)
        if not diagonal_only:
            B = batch_hessian_dict["B1"]["B2"][batch, :].view(m * n, n * p)
            C = batch_hessian_dict["B2"]["B1"][batch, :].view(n * p, m * n)
        else:
            B = torch.zeros(m * n, n * p)
            C = torch.zeros(n * p, m * n)
        D = batch_hessian_dict["B2"]["B2"][batch, :].view(n * p, n * p)

        # Assemble expected Hessian matrix from blocks
        row_0 = torch.cat((A, B), dim=1)
        row_1 = torch.cat((C, D), dim=1)
        expected_hessian_matrix = torch.cat((row_0, row_1), dim=0)

        assert torch.all(batch_hessian_matrix[batch, :] == expected_hessian_matrix), err_str
