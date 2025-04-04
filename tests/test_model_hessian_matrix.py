"""Test code for `model_hessian_matrix()`."""

# pylint: disable=invalid-name

import pytest
import torch
from torch import nn

from src.hesse import model_hessian_matrix
from tests.conftest import commutation_matrix, randint


def test_model_hessian_matrix_bilinear(bilinear: nn.Module) -> None:
    """Test with bilinear model."""
    # Make input data
    x1 = randint((bilinear.B.in1_features,))
    x2 = randint((bilinear.B.in2_features,))
    inputs = (x1, x2)

    # Compute Hessian matrix
    # - PyTorch issues performance warning for unimplemented batching rule
    # - This does not affect the correctness of the implementation.
    with pytest.warns(UserWarning):
        hessian_matrix = model_hessian_matrix(model=bilinear, inputs=inputs)

    # Check Hessian matrix shape
    err_str = "Error in Hessian matrix shape"
    expected_shape = 2 * torch.Size([bilinear.B.weight.numel()])
    assert hessian_matrix.shape == expected_shape, err_str

    # Check Hessian matrix values
    err_str = "Error in Hessian matrix values"
    assert torch.all(hessian_matrix == 0.0), err_str


@pytest.mark.parametrize("diagonal_only", [True, False])
def test_model_hessian_dict_double_bilinear(
    double_bilinear: nn.Module, diagonal_only: bool
) -> None:
    """Test with double-bilinear model."""
    # Make aliases for brevity
    B1 = double_bilinear.B1
    B2 = double_bilinear.B2
    m, n, p = B1.shape[0], B1.shape[1], B2.shape[1]

    # Make input data
    x1 = randint((m,))
    x2 = randint((p,))
    inputs = (x1, x2)

    # Compute Hessian matrix
    hessian_matrix = model_hessian_matrix(
        model=double_bilinear,
        inputs=inputs,
        diagonal_only=diagonal_only,
    )

    # Check Hessian matrix shape
    err_str = "Error in Hessian matrix shape"
    expected_shape = 2 * torch.Size([(m * n) + (n * p)])
    assert hessian_matrix.shape == expected_shape, err_str

    # Check Hessian matrix values
    # - See comments in `test_model_hessian_dict.py` for derivations
    err_str = "Error in Hessian matrix values"
    K = commutation_matrix(m, n)

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
