"""Test code for batch `model_hessian_dict()`."""

# pylint: disable=invalid-name

import pytest
import torch
from torch import nn

from hesse import model_hessian_dict
from tests.conftest import commutation_matrix, randint


def test_batch_model_hessian_dict_bilinear(bilinear: nn.Module, batch_size: int) -> None:
    """Batch `model_hessian_dict()` with bilinear model."""
    # Make batch inputs
    x1 = randint((batch_size, bilinear.B.in1_features))
    x2 = randint((batch_size, bilinear.B.in2_features))
    batch_inputs = (x1, x2)

    # Compute batch Hessian dict
    # - PyTorch issues performance warning for unimplemented batching rule
    # - This does not affect the correctness of the implementation.
    with pytest.warns(UserWarning):
        batch_hessian_dict = model_hessian_dict(
            model=bilinear,
            inputs=batch_inputs,
        )

    # Check keys
    err_str = "Key error"
    assert list(batch_hessian_dict.keys()) == ["B.weight"], err_str
    assert list(batch_hessian_dict["B.weight"].keys()) == ["B.weight"], err_str

    # Check batch Hessian shapes
    err_str = "Error in batch Hessian shapes"
    output_shape = torch.Size([batch_size, 1])  # Model output is (batch_size, 1)
    expected_shape = output_shape + 2 * bilinear.B.weight.shape
    assert batch_hessian_dict["B.weight"]["B.weight"].shape == expected_shape

    # Check Hessian values
    err_str = "Error in Hessian values"
    for batch in range(batch_size):
        assert torch.all(batch_hessian_dict["B.weight"]["B.weight"][batch] == 0.0), err_str


@pytest.mark.parametrize("diagonal_only", [True, False])
def test_batch_model_hessian_dict_double_bilinear(
    double_bilinear: nn.Module, batch_size: int, diagonal_only: bool
) -> None:
    """Batch `model_hessian_dict()` with double-bilinear model."""
    # Make aliases for brevity
    B1 = double_bilinear.B1
    B2 = double_bilinear.B2
    m, n, p = B1.shape[0], B1.shape[1], B2.shape[1]

    # Make batch inputs
    x1 = randint((batch_size, m))
    x2 = randint((batch_size, p))
    batch_inputs = (x1, x2)

    # Compute batch Hessian dict
    batch_hessian_dict = model_hessian_dict(
        model=double_bilinear,
        inputs=batch_inputs,
        diagonal_only=diagonal_only,
    )

    # Check keys
    err_str = "Key error"
    assert list(batch_hessian_dict.keys()) == ["B1", "B2"], err_str
    if not diagonal_only:
        assert list(batch_hessian_dict["B1"].keys()) == ["B1", "B2"], err_str
        assert list(batch_hessian_dict["B2"].keys()) == ["B1", "B2"], err_str
    else:
        assert list(batch_hessian_dict["B1"].keys()) == ["B1"], err_str
        assert list(batch_hessian_dict["B2"].keys()) == ["B2"], err_str

    # Check batch Hessian shapes
    err_str = "Error in batch Hessian shapes"
    output_shape = torch.Size([batch_size])  # Model output is (batch_size)

    # Check (B1, B1) Hessian shape
    expected_shape = output_shape + 2 * B1.shape
    assert batch_hessian_dict["B1"]["B1"].shape == expected_shape, err_str

    # Check (B1, B2) Hessian shape
    if not diagonal_only:
        expected_shape = output_shape + B1.shape + B2.shape
        assert batch_hessian_dict["B1"]["B2"].shape == expected_shape, err_str

    # Check (B2, B1) Hessian shape
    if not diagonal_only:
        expected_shape = output_shape + B2.shape + B1.shape
        assert batch_hessian_dict["B2"]["B1"].shape == expected_shape, err_str

    # Check (B2, B2) Hessian shape
    expected_shape = output_shape + 2 * B2.shape
    assert batch_hessian_dict["B2"]["B2"].shape == expected_shape, err_str

    # Check Hessian values
    # - See `test_model_hessian_dict.py` for derivations
    err_str = "Error in Hessian values"
    K = commutation_matrix(m, n)

    # Check (B1, B1) Hessian values
    assert torch.all(batch_hessian_dict["B1"]["B1"] == 0.0), err_str

    # Check (B1, B2) Hessian values
    if not diagonal_only:
        for batch in range(batch_size):
            actual_value = batch_hessian_dict["B1"]["B2"][batch].view(m * n, n * p)
            expected_value = K @ torch.kron(torch.eye(n), torch.outer(x1[batch], x2[batch]))
            assert actual_value.equal(expected_value), err_str

    # Check (B2, B1) Hessian values
    if not diagonal_only:
        for batch in range(batch_size):
            actual_value = batch_hessian_dict["B2"]["B1"][batch].view(n * p, m * n)
            expected_value = torch.kron(torch.eye(n), torch.outer(x2[batch], x1[batch])) @ K.T
            assert actual_value.equal(expected_value), err_str

    # Check (B2, B2) Hessian values
    assert torch.all(batch_hessian_dict["B2"]["B2"] == 0.0), err_str


def test_batch_model_hessian_dict_double_bilinear_frozen(
    double_bilinear_frozen: nn.Module, batch_size: int
) -> None:
    """Batch `model_hessian_dict()` with frozen double-bilinear model."""
    # Make aliases for brevity
    B1 = double_bilinear_frozen.B1
    B2 = double_bilinear_frozen.B2
    m, p = B1.shape[0], B2.shape[1]

    # Make batch inputs
    x1 = randint((batch_size, m))
    x2 = randint((batch_size, p))
    batch_inputs = (x1, x2)

    # Compute batch Hessian dict
    batch_hessian_dict = model_hessian_dict(
        model=double_bilinear_frozen,
        inputs=batch_inputs,
    )

    # Check keys
    err_str = "Key error"
    assert list(batch_hessian_dict.keys()) == ["B2"], err_str
    assert list(batch_hessian_dict["B2"].keys()) == ["B2"], err_str

    # Check batch Hessian shapes
    err_str = "Error in batch Hessian shapes"

    # Check (B2, B2) Hessian shape
    expected_shape = torch.Size([batch_size]) + 2 * B2.shape
    assert batch_hessian_dict["B2"]["B2"].shape == expected_shape, err_str

    # Check Hessian values
    err_str = "Error in Hessian values"

    # Check (B2, B2) Hessian values
    assert torch.all(batch_hessian_dict["B2"]["B2"] == 0.0), err_str


@pytest.mark.parametrize("diagonal_only", [True, False])
def test_batch_model_hessian_dict_sum_norms_squared(
    sum_norms_squared: nn.Module, batch_size: int, diagonal_only: bool
) -> None:
    """Batch `model_hessian_dict()` with sum-norms-squared model."""
    # Make aliases for brevity
    A1 = sum_norms_squared.A1
    A2 = sum_norms_squared.A2
    m, n = A1.shape[0], A1.shape[1]

    # Make batch inputs
    batch_inputs = randint((batch_size,))  # Model output is (batch_size)

    # Compute batch Hessian dict
    batch_hessian_dict = model_hessian_dict(
        model=sum_norms_squared,
        inputs=batch_inputs,
        diagonal_only=diagonal_only,
    )

    # Check keys
    err_str = "Key error"
    assert list(batch_hessian_dict.keys()) == ["A1", "A2"], err_str
    if not diagonal_only:
        assert list(batch_hessian_dict["A1"].keys()) == ["A1", "A2"], err_str
        assert list(batch_hessian_dict["A2"].keys()) == ["A1", "A2"], err_str
    else:
        assert list(batch_hessian_dict["A1"].keys()) == ["A1"], err_str
        assert list(batch_hessian_dict["A2"].keys()) == ["A2"], err_str

    # Check batch Hessian shapes
    err_str = "Error in batch Hessian shapes"
    output_shape = torch.Size([batch_size])  # Model output is (batch_size)

    # Check (A1, A1) Hessian shape
    expected_shape = output_shape + 2 * A1.shape
    assert batch_hessian_dict["A1"]["A1"].shape == expected_shape, err_str

    # Check (A1, A2) Hessian shape
    if not diagonal_only:
        expected_shape = output_shape + A1.shape + A2.shape
        assert batch_hessian_dict["A1"]["A2"].shape == expected_shape, err_str

    # Check (A2, A1) Hessian shape
    if not diagonal_only:
        expected_shape = output_shape + A2.shape + A1.shape
        assert batch_hessian_dict["A2"]["A1"].shape == expected_shape, err_str

    # Check (A2, A2) Hessian shape
    expected_shape = output_shape + 2 * A2.shape
    assert batch_hessian_dict["A2"]["A2"].shape == expected_shape, err_str

    # Check Hessian values
    # - See `test_model_hessian_dict.py` for derivations
    err_str = "Error in Hessian values"

    # Check (A1, A1) Hessian values
    for batch in range(batch_size):
        actual_value = batch_hessian_dict["A1"]["A1"][batch].view(m * n, m * n)
        expected_value = 2.0 * batch_inputs[batch] * torch.eye(m * n)
        assert actual_value.equal(expected_value), err_str

    # Check (A1, A2) Hessian values
    if not diagonal_only:
        assert torch.all(batch_hessian_dict["A1"]["A2"] == 0.0), err_str

    # Check (A2, A1) Hessian values
    if not diagonal_only:
        assert torch.all(batch_hessian_dict["A2"]["A1"] == 0.0), err_str

    # Check (A2, A2) Hessian values
    for batch in range(batch_size):
        actual_value = batch_hessian_dict["A2"]["A2"][batch].view(m * n, m * n)
        expected_value = 2.0 * batch_inputs[batch] * torch.eye(m * n)
        assert actual_value.equal(expected_value), err_str


def test_batch_model_hessian_dict_sum_norms_squared_frozen(
    sum_norms_squared_frozen: nn.Module, batch_size: int
) -> None:
    """Batch `model_hessian_dict()` with frozen sum-norms-squared model."""
    # Make aliases for brevity
    A1 = sum_norms_squared_frozen.A1
    A2 = sum_norms_squared_frozen.A2
    m, n = A1.shape[0], A1.shape[1]

    # Make batch inputs
    batch_inputs = randint((batch_size,))  # Model output is (batch_size)

    # Compute batch Hessian dict
    batch_hessian_dict = model_hessian_dict(
        model=sum_norms_squared_frozen,
        inputs=batch_inputs,
    )

    # Check keys
    err_str = "Key error"
    assert list(batch_hessian_dict.keys()) == ["A2"], err_str
    assert list(batch_hessian_dict["A2"].keys()) == ["A2"], err_str

    # Check batch Hessian shapes
    err_str = "Error in batch Hessian shapes"

    # Check (A2, A2) Hessian shape
    expected_shape = torch.Size([batch_size]) + 2 * A2.shape
    assert batch_hessian_dict["A2"]["A2"].shape == expected_shape, err_str

    # Check Hessian values
    err_str = "Error in Hessian values"

    # Check (A2, A2) Hessian values
    for batch in range(batch_size):
        actual_value = batch_hessian_dict["A2"]["A2"][batch].view(m * n, m * n)
        expected_value = 2.0 * batch_inputs[batch] * torch.eye(m * n)
        assert actual_value.equal(expected_value), err_str
