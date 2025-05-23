"""Test code for non-batch `model_hessian_matrix()`."""

# pylint: disable=invalid-name

import pytest
import torch
from torch import nn

from hesse import model_hessian_matrix
from tests.conftest import commutation_matrix, randint


def test_model_hessian_matrix_bilinear(bilinear: nn.Module) -> None:
    """Non-batch `model_hessian_matrix()` with bilinear model."""
    # Make inputs
    x1 = randint((bilinear.B.in1_features,))
    x2 = randint((bilinear.B.in2_features,))
    inputs = (x1, x2)

    # Compute Hessian matrix
    # - PyTorch issues performance warning for unimplemented batching rule
    # - This does not affect the correctness of the implementation.
    with pytest.warns(UserWarning):
        hessian_matrix = model_hessian_matrix(
            model=bilinear,
            inputs=inputs,
            is_batch=False,
        )

    # Check Hessian matrix shapes
    err_str = "Error in Hessian matrix shape"
    output_shape = torch.Size([1])  # Model output is (1)
    expected_shape = output_shape + 2 * torch.Size([bilinear.B.weight.numel()])
    assert hessian_matrix.shape == expected_shape, err_str

    # Check Hessian values
    err_str = "Error in Hessian values"
    assert torch.all(hessian_matrix == 0.0), err_str


@pytest.mark.parametrize("diagonal_only", [True, False])
def test_model_hessian_matrix_double_bilinear(
    double_bilinear: nn.Module, diagonal_only: bool
) -> None:
    """Non-batch `model_hessian_matrix()` with double-bilinear model."""
    # Make aliases for brevity
    B1 = double_bilinear.B1
    B2 = double_bilinear.B2
    m, n, p = B1.shape[0], B1.shape[1], B2.shape[1]

    # Make inputs
    x1 = randint((m,))
    x2 = randint((p,))
    inputs = (x1, x2)

    # Compute Hessian matrix
    hessian_matrix = model_hessian_matrix(
        model=double_bilinear,
        inputs=inputs,
        diagonal_only=diagonal_only,
        is_batch=False,
    )

    # Check Hessian matrix shape
    err_str = "Error in Hessian matrix shape"
    expected_shape = 2 * torch.Size([(m * n) + (n * p)])
    assert hessian_matrix.shape == expected_shape, err_str

    # Check Hessian values
    # - See comments in `test_model_hessian_dict.py` for derivations
    err_str = "Error in Hessian values"
    K = commutation_matrix(m, n)

    hess_B1_B1 = hessian_matrix[: (m * n), : (m * n)]
    hess_B1_B2 = hessian_matrix[: (m * n), (m * n) :]
    hess_B2_B1 = hessian_matrix[(m * n) :, : (m * n)]
    hess_B2_B2 = hessian_matrix[(m * n) :, (m * n) :]

    # Check (B1, B1) Hessian values
    assert torch.all(hess_B1_B1 == 0.0), err_str

    # Check (B1, B2) Hessian values
    expected_value = 0.0
    if not diagonal_only:
        outer_prod = torch.outer(x1, x2)
        expected_value = K @ torch.kron(torch.eye(n), outer_prod)
    assert torch.all(hess_B1_B2 == expected_value), err_str

    # Check (B2, B1) Hessian values
    expected_value = 0.0
    if not diagonal_only:
        outer_prod = torch.outer(x2, x1)
        expected_value = torch.kron(torch.eye(n), outer_prod) @ K.T
    assert torch.all(hess_B2_B1 == expected_value), err_str

    # Check (B2, B2) Hessian values
    assert torch.all(hess_B2_B2 == 0.0), err_str
