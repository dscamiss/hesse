"""Test code for batch `model_hessian_matrix()`."""

# pylint: disable=invalid-name

import pytest
import torch
from torch import nn

from src.hesse import model_hessian_matrix
from tests.conftest import commutation_matrix, randint


def test_batch_model_hessian_matrix_bilinear(bilinear: nn.Module, batch_size: int) -> None:
    """Batch `model_hessian_matrix()` with bilinear model."""
    # Make batch inputs
    x1 = randint((batch_size, bilinear.B.in1_features))
    x2 = randint((batch_size, bilinear.B.in2_features))
    batch_inputs = (x1, x2)

    # Compute batch Hessian matrix
    # - PyTorch issues performance warning for unimplemented batching rule
    # - This does not affect the correctness of the implementation.
    with pytest.warns(UserWarning):
        batch_hessian_matrix = model_hessian_matrix(
            model=bilinear,
            inputs=batch_inputs,
        )

    # Check batch Hessian shape
    err_str = "Error in batch Hessian matrix shape"
    output_shape = torch.Size([batch_size, 1])  # Model output is (batch_size, 1)
    expected_shape = output_shape + 2 * torch.Size([bilinear.B.weight.numel()])
    assert batch_hessian_matrix.shape == expected_shape, err_str

    # Check Hessian values
    err_str = "Error in Hessian values"
    assert torch.all(batch_hessian_matrix == 0.0), err_str


@pytest.mark.parametrize("diagonal_only", [True, False])
def test_batch_model_hessian_dict_double_bilinear(
    double_bilinear: nn.Module, batch_size: int, diagonal_only: bool
) -> None:
    """Batch `model_hessian_matrix()` with double-bilinear model."""
    # Make aliases for brevity
    B1 = double_bilinear.B1
    B2 = double_bilinear.B2
    m, n, p = B1.shape[0], B1.shape[1], B2.shape[1]

    # Make batch inputs
    x1 = randint((batch_size, m))
    x2 = randint((batch_size, p))
    batch_inputs = (x1, x2)

    # Compute batch Hessian matrix
    batch_hessian_matrix = model_hessian_matrix(
        model=double_bilinear,
        inputs=batch_inputs,
        diagonal_only=diagonal_only,
    )

    # Check batch Hessian matrix shape
    err_str = "Error in batch Hessian matrix shape"
    expected_shape = torch.Size([batch_size]) + 2 * torch.Size([(m * n) + (n * p)])
    assert batch_hessian_matrix.shape == expected_shape, err_str

    # Check Hessian values
    # - See comments in `test_model_hessian_dict.py` for derivations
    err_str = "Error in Hessian values"
    K = commutation_matrix(m, n)

    for batch in range(batch_size):
        hess_B1_B1 = batch_hessian_matrix[batch, : (m * n), : (m * n)]
        hess_B1_B2 = batch_hessian_matrix[batch, : (m * n), (m * n) :]
        hess_B2_B1 = batch_hessian_matrix[batch, (m * n) :, : (m * n)]
        hess_B2_B2 = batch_hessian_matrix[batch, (m * n) :, (m * n) :]

        # Check (B1, B1) Hessian matrix values
        assert torch.all(hess_B1_B1 == 0.0), err_str

        # Check (B1, B2) Hessian matrix values
        expected_value = 0.0
        if not diagonal_only:
            outer_prod = torch.outer(x1[batch, :], x2[batch, :])
            expected_value = K @ torch.kron(torch.eye(n), outer_prod)
        assert torch.all(hess_B1_B2 == expected_value), err_str

        # Check (B2, B1) Hessian matrix values
        expected_value = 0.0
        if not diagonal_only:
            outer_prod = torch.outer(x2[batch, :], x1[batch, :])
            expected_value = torch.kron(torch.eye(n), outer_prod) @ K.T
        assert torch.all(hess_B2_B1 == expected_value), err_str

        # Check (B2, B2) Hessian matrix values
        assert torch.all(hess_B2_B2 == 0.0), err_str
