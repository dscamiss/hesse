"""Test code for `compute_batch_hessian()`."""

# pylint: disable=invalid-name

from typing import Callable

import pytest
import torch
from torch import nn

from src.hesse import compute_batch_hessian


@pytest.fixture(name="batch_size")
def fixture_batch_size() -> int:
    """Batch size for input data."""
    return 16


def test_compute_batch_hessian_bilinear(bilinear: nn.Module, batch_size: int) -> None:
    """Test `compute_batch_hessian()` with bilinear model."""
    # Make input data
    x1 = torch.randn(batch_size, bilinear.B.in1_features).requires_grad_(False)
    x2 = torch.randn(batch_size, bilinear.B.in2_features).requires_grad_(False)

    # PyTorch issues performance warning for unimplemented batching rule
    # - This does not affect the correctness of the implementation.
    with pytest.warns(UserWarning):
        hess = compute_batch_hessian(bilinear, x1, x2)

    # Check Hessian shape
    err_str = "Error in Hessian shape"
    output_shape = torch.Size([batch_size, 1])  # Model output is [batch_size, 1]
    expected_shape = output_shape + 2 * bilinear.B.weight.shape
    assert hess["B.weight"]["B.weight"].shape == expected_shape

    # Check Hessian values
    err_str = "Error in Hessian values"
    for batch in range(batch_size):
        assert torch.all(hess["B.weight"]["B.weight"][batch] == 0.0), err_str


def test_compute_batch_hessian_double_bilinear(
    double_bilinear: nn.Module, commutation_matrix: Callable, batch_size: int
) -> None:
    """Test `compute_batch_hessian()` with double-bilinear model."""
    # Make aliases for brevity
    B1 = double_bilinear.B1
    B2 = double_bilinear.B2
    m, n, p = B1.shape[0], B1.shape[1], B2.shape[1]

    # Make input data
    x1 = torch.randn(batch_size, m).requires_grad_(False)
    x2 = torch.randn(batch_size, p).requires_grad_(False)

    # Compute Hessian
    hess = compute_batch_hessian(double_bilinear, x1, x2)

    # Check Hessian shapes
    err_str = "Error in Hessian shape"
    output_shape = torch.Size([batch_size])  # Model output is [batch_size]

    # Check (B1, B1) Hessian shape
    expected_shape = output_shape + 2 * B1.shape
    assert hess["B1"]["B1"].shape == expected_shape, err_str

    # Check (B1, B2) Hessian shape
    expected_shape = output_shape + B1.shape + B2.shape
    assert hess["B1"]["B2"].shape == expected_shape, err_str

    # Check (B2, B1) Hessian shape
    expected_shape = output_shape + B2.shape + B1.shape
    assert hess["B2"]["B1"].shape == expected_shape, err_str

    # Check (B2, B2) Hessian shape
    expected_shape = output_shape + 2 * B2.shape
    assert hess["B2"]["B2"].shape == expected_shape, err_str

    # Check Hessian values
    # - Hessian calculations are in `test_compute_hessian.py`
    err_str = "Error in Hessian values"
    K = commutation_matrix(m, n).requires_grad_(False)

    # Check (B1, B1) Hessian values
    assert torch.all(hess["B1"]["B1"] == 0.0), err_str

    for batch in range(batch_size):
        # Check (B1, B2) Hessian values
        expected_value = K @ torch.kron(torch.eye(n), torch.outer(x1[batch], x2[batch]))
        assert torch.allclose(hess["B1"]["B2"][batch].view(m * n, n * p), expected_value), err_str

        # Check (B2, B1) Hessian values
        expected_value = torch.kron(torch.eye(n), torch.outer(x2[batch], x1[batch])) @ K.T
        assert torch.allclose(hess["B2"]["B1"][batch].view(n * p, m * n), expected_value), err_str

    # Check (B2, B2) Hessian values
    assert torch.all(hess["B2"]["B2"] == 0.0), err_str


def test_compute_batch_hessian_double_bilinear_frozen(
    double_bilinear_frozen: nn.Module, batch_size: int
) -> None:
    """Test `compute_batch_hessian()` with frozen double-bilinear model."""
    # Make aliases for brevity
    B1 = double_bilinear_frozen.B1
    B2 = double_bilinear_frozen.B2
    m, p = B1.shape[0], B2.shape[1]

    # Make input data
    x1 = torch.randn(batch_size, m).requires_grad_(False)
    x2 = torch.randn(batch_size, p).requires_grad_(False)

    # Compute Hessian
    hess = compute_batch_hessian(double_bilinear_frozen, x1, x2)

    # Check keys
    err_str = "Key error"
    assert list(hess.keys()) == ["B2"], err_str
    assert list(hess["B2"].keys()) == ["B2"], err_str

    # Check Hessian shapes
    err_str = "Error in Hessian shape"

    # Check (B2, B2) Hessian shape
    expected_shape = torch.Size([batch_size]) + 2 * B2.shape
    assert hess["B2"]["B2"].shape == expected_shape, err_str

    # Check Hessian values
    err_str = "Error in Hessian values"

    # Check (B2, B2) Hessian values
    assert torch.all(hess["B2"]["B2"] == 0.0), err_str


def test_compute_batch_hessian_sum_norms_squared(
    sum_norms_squared: nn.Module, batch_size: int
) -> None:
    """Test `compute_batch_hessian()` with sum-norms-squared model."""
    # Make aliases for brevity
    A1 = sum_norms_squared.A1
    A2 = sum_norms_squared.A2
    m, n = A1.shape[0], A1.shape[1]

    # Make input data
    x = torch.randn(batch_size).requires_grad_(False)

    # Compute Hessian
    hess = compute_batch_hessian(sum_norms_squared, x)

    # Check Hessian shapes
    err_str = "Error in Hessian shape"
    output_shape = torch.Size([batch_size])  # Model output is [batch_size]

    # Check (A1, A1) Hessian shape
    expected_shape = output_shape + 2 * A1.shape
    assert hess["A1"]["A1"].shape == expected_shape, err_str

    # Check (A1, A2) Hessian shape
    expected_shape = output_shape + A1.shape + A2.shape
    assert hess["A1"]["A2"].shape == expected_shape, err_str

    # Check (A2, A1) Hessian shape
    expected_shape = output_shape + A2.shape + A1.shape
    assert hess["A2"]["A1"].shape == expected_shape, err_str

    # Check (A2, A2) Hessian shape
    expected_shape = output_shape + 2 * A2.shape
    assert hess["A2"]["A2"].shape == expected_shape, err_str

    # Check Hessian values
    # - Hessian calculations are in `test_compute_hessian.py`
    err_str = "Error in Hessian values"

    # Check (A1, A2) Hessian values
    assert torch.all(hess["A1"]["A2"] == 0.0), err_str

    # Check (A2, A1) Hessian values
    assert torch.all(hess["A2"]["A1"] == 0.0), err_str

    for batch in range(batch_size):
        # Check (A1, A1) Hessian values
        expected_value = 2.0 * x[batch] * torch.eye(m * n)
        assert torch.allclose(hess["A1"]["A1"][batch].view(m * n, m * n), expected_value), err_str

        # Check (A2, A2) Hessian values
        expected_value = 2.0 * x[batch] * torch.eye(m * n)
        assert torch.allclose(hess["A2"]["A2"][batch].view(m * n, m * n), expected_value), err_str


def test_compute_batch_hessian_sum_norms_squared_frozen(
    sum_norms_squared_frozen: nn.Module, batch_size: int
) -> None:
    """Test `compute_batch_hessian()` with frozen sum-norms-squared model."""
    # Make aliases for brevity
    A1 = sum_norms_squared_frozen.A1
    A2 = sum_norms_squared_frozen.A2
    m, n = A1.shape[0], A1.shape[1]

    # Make input data
    x = torch.randn(batch_size).requires_grad_(False)

    # Compute Hessian
    hess = compute_batch_hessian(sum_norms_squared_frozen, x)

    # Check keys
    err_str = "Key error"
    assert list(hess.keys()) == ["A2"], err_str
    assert list(hess["A2"].keys()) == ["A2"], err_str

    # Check Hessian shapes
    err_str = "Error in Hessian shape"

    # Check (A2, A2) Hessian shape
    expected_shape = torch.Size([batch_size]) + 2 * A2.shape
    assert hess["A2"]["A2"].shape == expected_shape, err_str

    # Check Hessian values
    err_str = "Error in Hessian values"

    # Check (A2, A2) Hessian values
    for batch in range(batch_size):
        expected_value = 2.0 * x[batch] * torch.eye(m * n)
        assert torch.allclose(hess["A2"]["A2"][batch].view(m * n, m * n), expected_value), err_str
