"""Test code for `batch_model_hessian_dict()`."""

# pylint: disable=invalid-name

from typing import Callable

import pytest
import torch
from torch import nn

from src.hesse import batch_model_hessian_dict


def test_batch_model_hessian_dict_bilinear(bilinear: nn.Module, batch_size: int) -> None:
    """Test with bilinear model."""
    # Make input data
    x1 = torch.randn(batch_size, bilinear.B.in1_features).requires_grad_(False)
    x2 = torch.randn(batch_size, bilinear.B.in2_features).requires_grad_(False)
    batch_inputs = (x1, x2)

    # Compute Hessian dict
    # - PyTorch issues performance warning for unimplemented batching rule
    # - This does not affect the correctness of the implementation.
    with pytest.warns(UserWarning):
        hessian_dict = batch_model_hessian_dict(
            model=bilinear,
            batch_inputs=batch_inputs,
        )

    # Check keys
    err_str = "Key error"
    assert list(hessian_dict.keys()) == ["B.weight"], err_str
    assert list(hessian_dict["B.weight"].keys()) == ["B.weight"], err_str

    # Check Hessian shapes
    err_str = "Error in Hessian shapes"
    output_shape = torch.Size([batch_size, 1])  # Model output is [batch_size, 1]
    expected_shape = output_shape + 2 * bilinear.B.weight.shape
    assert hessian_dict["B.weight"]["B.weight"].shape == expected_shape

    # Check Hessian values
    err_str = "Error in Hessian values"
    for batch in range(batch_size):
        assert torch.all(hessian_dict["B.weight"]["B.weight"][batch] == 0.0), err_str


@pytest.mark.parametrize("diagonal_only", [True, False])
def test_batch_model_hessian_dict_double_bilinear(
    double_bilinear: nn.Module, commutation_matrix: Callable, batch_size: int, diagonal_only: bool
) -> None:
    """Test with double-bilinear model."""
    # Make aliases for brevity
    B1 = double_bilinear.B1
    B2 = double_bilinear.B2
    m, n, p = B1.shape[0], B1.shape[1], B2.shape[1]

    # Make input data
    x1 = torch.randn(batch_size, m).requires_grad_(False)
    x2 = torch.randn(batch_size, p).requires_grad_(False)
    batch_inputs = (x1, x2)

    # Compute Hessian dict
    hessian_dict = batch_model_hessian_dict(
        model=double_bilinear,
        batch_inputs=batch_inputs,
        diagonal_only=diagonal_only,
    )

    # Check keys
    err_str = "Key error"
    assert list(hessian_dict.keys()) == ["B1", "B2"], err_str
    if not diagonal_only:
        assert list(hessian_dict["B1"].keys()) == ["B1", "B2"], err_str
        assert list(hessian_dict["B2"].keys()) == ["B1", "B2"], err_str
    else:
        assert list(hessian_dict["B1"].keys()) == ["B1"], err_str
        assert list(hessian_dict["B2"].keys()) == ["B2"], err_str

    # Check Hessian shapes
    err_str = "Error in Hessian shapes"
    output_shape = torch.Size([batch_size])  # Model output is [batch_size]

    # Check (B1, B1) Hessian shape
    expected_shape = output_shape + 2 * B1.shape
    assert hessian_dict["B1"]["B1"].shape == expected_shape, err_str

    # Check (B1, B2) Hessian shape
    if not diagonal_only:
        expected_shape = output_shape + B1.shape + B2.shape
        assert hessian_dict["B1"]["B2"].shape == expected_shape, err_str

    # Check (B2, B1) Hessian shape
    if not diagonal_only:
        expected_shape = output_shape + B2.shape + B1.shape
        assert hessian_dict["B2"]["B1"].shape == expected_shape, err_str

    # Check (B2, B2) Hessian shape
    expected_shape = output_shape + 2 * B2.shape
    assert hessian_dict["B2"]["B2"].shape == expected_shape, err_str

    # Check Hessian values
    # - Hessian calculations are in `test_model_hessian.py`
    err_str = "Error in Hessian values"
    K = commutation_matrix(m, n).requires_grad_(False)

    # Check (B1, B1) Hessian values
    assert torch.all(hessian_dict["B1"]["B1"] == 0.0), err_str

    # Check (B1, B2) Hessian values
    if not diagonal_only:
        for batch in range(batch_size):
            actual_value = hessian_dict["B1"]["B2"][batch].view(m * n, n * p)
            expected_value = K @ torch.kron(torch.eye(n), torch.outer(x1[batch], x2[batch]))
            assert torch.allclose(actual_value, expected_value), err_str

    # Check (B2, B1) Hessian values
    if not diagonal_only:
        for batch in range(batch_size):
            actual_value = hessian_dict["B2"]["B1"][batch].view(n * p, m * n)
            expected_value = torch.kron(torch.eye(n), torch.outer(x2[batch], x1[batch])) @ K.T
            assert torch.allclose(actual_value, expected_value), err_str

    # Check (B2, B2) Hessian values
    assert torch.all(hessian_dict["B2"]["B2"] == 0.0), err_str


def test_batch_model_hessian_dict_double_bilinear_frozen(
    double_bilinear_frozen: nn.Module, batch_size: int
) -> None:
    """Test with frozen double-bilinear model."""
    # Make aliases for brevity
    B1 = double_bilinear_frozen.B1
    B2 = double_bilinear_frozen.B2
    m, p = B1.shape[0], B2.shape[1]

    # Make input data
    x1 = torch.randn(batch_size, m).requires_grad_(False)
    x2 = torch.randn(batch_size, p).requires_grad_(False)
    batch_inputs = (x1, x2)

    # Compute Hessian dict
    hessian_dict = batch_model_hessian_dict(
        model=double_bilinear_frozen,
        batch_inputs=batch_inputs,
    )

    # Check keys
    err_str = "Key error"
    assert list(hessian_dict.keys()) == ["B2"], err_str
    assert list(hessian_dict["B2"].keys()) == ["B2"], err_str

    # Check Hessian shapes
    err_str = "Error in Hessian shapes"

    # Check (B2, B2) Hessian shape
    expected_shape = torch.Size([batch_size]) + 2 * B2.shape
    assert hessian_dict["B2"]["B2"].shape == expected_shape, err_str

    # Check Hessian values
    err_str = "Error in Hessian values"

    # Check (B2, B2) Hessian values
    assert torch.all(hessian_dict["B2"]["B2"] == 0.0), err_str


@pytest.mark.parametrize("diagonal_only", [True, False])
def test_batch_model_hessian_dict_sum_norms_squared(
    sum_norms_squared: nn.Module, batch_size: int, diagonal_only: bool
) -> None:
    """Test with sum-norms-squared model."""
    # Make aliases for brevity
    A1 = sum_norms_squared.A1
    A2 = sum_norms_squared.A2
    m, n = A1.shape[0], A1.shape[1]

    # Make input data
    batch_inputs = torch.randn(batch_size).requires_grad_(False)

    # Compute Hessian dict
    hessian_dict = batch_model_hessian_dict(
        model=sum_norms_squared,
        batch_inputs=batch_inputs,
        diagonal_only=diagonal_only,
    )

    # Check keys
    err_str = "Key error"
    assert list(hessian_dict.keys()) == ["A1", "A2"], err_str
    if not diagonal_only:
        assert list(hessian_dict["A1"].keys()) == ["A1", "A2"], err_str
        assert list(hessian_dict["A2"].keys()) == ["A1", "A2"], err_str
    else:
        assert list(hessian_dict["A1"].keys()) == ["A1"], err_str
        assert list(hessian_dict["A2"].keys()) == ["A2"], err_str

    # Check Hessian shapes
    err_str = "Error in Hessian shapes"
    output_shape = torch.Size([batch_size])  # Model output is [batch_size]

    # Check (A1, A1) Hessian shape
    expected_shape = output_shape + 2 * A1.shape
    assert hessian_dict["A1"]["A1"].shape == expected_shape, err_str

    # Check (A1, A2) Hessian shape
    if not diagonal_only:
        expected_shape = output_shape + A1.shape + A2.shape
        assert hessian_dict["A1"]["A2"].shape == expected_shape, err_str

    # Check (A2, A1) Hessian shape
    if not diagonal_only:
        expected_shape = output_shape + A2.shape + A1.shape
        assert hessian_dict["A2"]["A1"].shape == expected_shape, err_str

    # Check (A2, A2) Hessian shape
    expected_shape = output_shape + 2 * A2.shape
    assert hessian_dict["A2"]["A2"].shape == expected_shape, err_str

    # Check Hessian values
    # - Hessian calculations are in `test_model_hessian.py`
    err_str = "Error in Hessian values"

    # Check (A1, A1) Hessian values
    for batch in range(batch_size):
        actual_value = hessian_dict["A1"]["A1"][batch].view(m * n, m * n)
        expected_value = 2.0 * batch_inputs[batch] * torch.eye(m * n)
        assert torch.allclose(actual_value, expected_value), err_str

    # Check (A1, A2) Hessian values
    if not diagonal_only:
        assert torch.all(hessian_dict["A1"]["A2"] == 0.0), err_str

    # Check (A2, A1) Hessian values
    if not diagonal_only:
        assert torch.all(hessian_dict["A2"]["A1"] == 0.0), err_str

    # Check (A2, A2) Hessian values
    for batch in range(batch_size):
        actual_value = hessian_dict["A2"]["A2"][batch].view(m * n, m * n)
        expected_value = 2.0 * batch_inputs[batch] * torch.eye(m * n)
        assert torch.allclose(actual_value, expected_value), err_str


def test_batch_model_hessian_dict_sum_norms_squared_frozen(
    sum_norms_squared_frozen: nn.Module, batch_size: int
) -> None:
    """Test with frozen sum-norms-squared model."""
    # Make aliases for brevity
    A1 = sum_norms_squared_frozen.A1
    A2 = sum_norms_squared_frozen.A2
    m, n = A1.shape[0], A1.shape[1]

    # Make input data
    batch_inputs = torch.randn(batch_size).requires_grad_(False)

    # Compute Hessian dict
    hessian_dict = batch_model_hessian_dict(
        model=sum_norms_squared_frozen,
        batch_inputs=batch_inputs,
    )

    # Check keys
    err_str = "Key error"
    assert list(hessian_dict.keys()) == ["A2"], err_str
    assert list(hessian_dict["A2"].keys()) == ["A2"], err_str

    # Check Hessian shapes
    err_str = "Error in Hessian shapes"

    # Check (A2, A2) Hessian shape
    expected_shape = torch.Size([batch_size]) + 2 * A2.shape
    assert hessian_dict["A2"]["A2"].shape == expected_shape, err_str

    # Check Hessian values
    err_str = "Error in Hessian values"

    # Check (A2, A2) Hessian values
    for batch in range(batch_size):
        actual_value = hessian_dict["A2"]["A2"][batch].view(m * n, m * n)
        expected_value = 2.0 * batch_inputs[batch] * torch.eye(m * n)
        assert torch.allclose(actual_value, expected_value), err_str
