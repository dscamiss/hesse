"""Test code for non-batch `model_hessian_dict()`."""

# pylint: disable=invalid-name

import pytest
import torch
from torch import nn

from src.hesse import model_hessian_dict
from tests.conftest import commutation_matrix, randint


def test_model_hessian_dict_bilinear(bilinear: nn.Module) -> None:
    """Non-batch `model_hessian_dict()` with bilinear model."""
    # Make inputs
    x1 = randint((bilinear.B.in1_features,))
    x2 = randint((bilinear.B.in2_features,))
    inputs = (x1, x2)

    # Compute Hessian dict
    # - PyTorch issues performance warning for unimplemented batching rule
    # - This does not affect the correctness of the implementation.
    with pytest.warns(UserWarning):
        hessian_dict = model_hessian_dict(
            model=bilinear,
            inputs=inputs,
            is_batched=False,
        )

    # Check keys
    err_str = "Key error"
    assert list(hessian_dict.keys()) == ["B.weight"], err_str
    assert list(hessian_dict["B.weight"].keys()) == ["B.weight"], err_str

    # Check Hessian shapes
    err_str = "Error in Hessian shapes"
    expected_shape = 2 * bilinear.B.weight.shape
    assert hessian_dict["B.weight"]["B.weight"][0].shape == expected_shape, err_str

    # Check Hessian values
    err_str = "Error in Hessian values"
    assert torch.all(hessian_dict["B.weight"]["B.weight"][0] == 0.0), err_str


@pytest.mark.parametrize("diagonal_only", [True, False])
def test_model_hessian_dict_double_bilinear(
    double_bilinear: nn.Module, diagonal_only: bool
) -> None:
    """Non-batch `model_hessian_dict()` with double-bilinear model."""
    # Make aliases for brevity
    B1 = double_bilinear.B1
    B2 = double_bilinear.B2
    m, n, p = B1.shape[0], B1.shape[1], B2.shape[1]

    # Make inputs
    x1 = randint((m,))
    x2 = randint((p,))
    inputs = (x1, x2)

    # Compute Hessian dict
    hessian_dict = model_hessian_dict(
        model=double_bilinear,
        inputs=inputs,
        diagonal_only=diagonal_only,
        is_batched=False,
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

    # Check (B1, B1) Hessian shape
    expected_shape = 2 * B1.shape
    assert hessian_dict["B1"]["B1"].shape == expected_shape, err_str

    # Check (B1, B2) Hessian shape
    if not diagonal_only:
        expected_shape = B1.shape + B2.shape
        assert hessian_dict["B1"]["B2"].shape == expected_shape, err_str

    # Check (B2, B1) Hessian shape
    if not diagonal_only:
        expected_shape = B2.shape + B1.shape
        assert hessian_dict["B2"]["B1"].shape == expected_shape, err_str

    # Check (B2, B2) Hessian shape
    expected_shape = 2 * B2.shape
    assert hessian_dict["B2"]["B2"].shape == expected_shape, err_str

    # Check Hessian values
    # - Note that
    #
    #       d^2 f(B1, B2) . [(V1, V2), (W1, W2)]
    #           = x^t W1 V2 y + x^t V1 W2 y
    #           = tr(x^t W1 V2 y) + tr(x^t V1 W2 y)
    #           = tr(V2 y x^t W1) + tr(W2 y x^t V1).
    #
    # - The first term is
    #
    #       tr(V2 y x^t W1)
    #           = vec(V2^t)^t (I (X) y x^t) vec(W1)
    #           = flat(V2)^t (I (X) y x^t) K^t flat(W1)
    #
    #   and the second term is
    #
    #       tr(W2 y x^t V1)
    #           = tr(V1^t x y^t W2^t)
    #           = vec(V1)^t (I (X) x y^t) vec(W2^t)
    #           = flat(V1)^t K (I (X) x y^t) flat(W2),
    #
    #   where
    #
    #   * tr() is the trace function,
    #   * vec() is the column-stacking vectorization map,
    #   * flat() is the row-stacking vectorization map,
    #   * (X) is the Kronecker product, and
    #   * K is the commutation matrix K_{n,p}.
    #
    # - This means that the Hessian blocks are
    #      Hess_{B1,B1} = <zeros>
    #      Hess_{B1,B2} = K (I (X) x y^t)
    #      Hess_{B2,B1} = (I (X) y x^t) K^t
    #      Hess_{B2,B2} = <zeros>.

    err_str = "Error in Hessian values"
    K = commutation_matrix(m, n)

    # Check (B1, B1) Hessian values
    assert torch.all(hessian_dict["B1"]["B1"] == 0.0), err_str

    # Check (B1, B2) Hessian values
    if not diagonal_only:
        actual_value = hessian_dict["B1"]["B2"].view(m * n, n * p)
        expected_value = K @ torch.kron(torch.eye(n), torch.outer(x1, x2))
        assert actual_value.equal(expected_value), err_str

    # Check (B2, B1) Hessian values
    if not diagonal_only:
        actual_value = hessian_dict["B2"]["B1"].view(n * p, m * n)
        expected_value = torch.kron(torch.eye(n), torch.outer(x2, x1)) @ K.T
        assert actual_value.equal(expected_value), err_str

    # Check (B2, B2) Hessian values
    assert torch.all(hessian_dict["B2"]["B2"] == 0.0), err_str


def test_model_hessian_dict_double_bilinear_frozen(double_bilinear_frozen: nn.Module) -> None:
    """Non-batch `model_hessian_dict()` with frozen double-bilinear model."""
    # Make aliases for brevity
    B1 = double_bilinear_frozen.B1
    B2 = double_bilinear_frozen.B2
    m, p = B1.shape[0], B2.shape[1]

    # Make inputs
    x1 = randint((m,))
    x2 = randint((p,))
    inputs = (x1, x2)

    # Compute Hessian dict
    hessian_dict = model_hessian_dict(
        model=double_bilinear_frozen,
        inputs=inputs,
        is_batched=False,
    )

    # Check keys
    err_str = "Key error"
    assert list(hessian_dict.keys()) == ["B2"], err_str
    assert list(hessian_dict["B2"].keys()) == ["B2"], err_str

    # Check Hessian shapes
    err_str = "Error in Hessian shapes"

    # Check (B2, B2) Hessian shape
    expected_shape = 2 * B2.shape
    assert hessian_dict["B2"]["B2"].shape == expected_shape, err_str

    # Check Hessian values
    err_str = "Error in Hessian values"

    # Check (B2, B2) Hessian values
    assert torch.all(hessian_dict["B2"]["B2"] == 0.0), err_str


@pytest.mark.parametrize("diagonal_only", [True, False])
def test_model_hessian_dict_sum_norms_squared(
    sum_norms_squared: nn.Module, diagonal_only: bool
) -> None:
    """Non-batch `model_hessian_dict()` with sum-norms-squared model."""
    # Make aliases for brevity
    A1 = sum_norms_squared.A1
    A2 = sum_norms_squared.A2
    m, n = A1.shape[0], A1.shape[1]

    # Make inputs
    inputs = randint()

    # Compute Hessian dict
    hessian_dict = model_hessian_dict(
        model=sum_norms_squared,
        inputs=inputs,
        diagonal_only=diagonal_only,
        is_batched=False,
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

    # Check (A1, A1) Hessian shape
    expected_shape = 2 * A1.shape
    assert hessian_dict["A1"]["A1"].shape == expected_shape, err_str

    # Check (A1, A2) Hessian shape
    if not diagonal_only:
        expected_shape = A1.shape + A2.shape
        assert hessian_dict["A1"]["A2"].shape == expected_shape, err_str

    # Check (A2, A1) Hessian shape
    if not diagonal_only:
        expected_shape = A2.shape + A1.shape
        assert hessian_dict["A2"]["A1"].shape == expected_shape, err_str

    # Check (A2, A2) Hessian shape
    expected_shape = 2 * A2.shape
    assert hessian_dict["A2"]["A2"].shape == expected_shape, err_str

    # Check Hessian values
    # - Note that
    #
    #       d^2 f(A1, A2) . [(V1, V2), (W1, W2)]
    #           = 2x tr(V1^t W1) + 2x tr(V2^t W2).
    #
    # - The first term is
    #
    #       2x tr(V1^t W1)
    #           = 2x vec(V1)^t vec(W1)
    #           = flat(V1)^t (2x K K^t) flat(W1)
    #           = flat(V1)^t (2x I) flat(W1)
    #
    #   and similarly the second term is
    #
    #       2x tr(V2^t W2)
    #           = flat(V2)^t (2x I) flat(W2),
    #
    #   where
    #
    #   * tr() is the trace function,
    #   * vec() is the column-stacking vectorization map,
    #   * flat() is the row-stacking vectorization map,
    #   * K is the commutation matrix K_{m, n}.
    #
    # - This means that the Hessian blocks are
    #      Hess_{A1,A1} = 2x I
    #      Hess_{A1,A2} = <zeros>
    #      Hess_{A2,A1} = <zeros>
    #      Hess_{A2,A2} = 2x I.

    err_str = "Error in Hessian values"

    # Check (A1, A1) Hessian values
    actual_value = hessian_dict["A1"]["A1"].view(m * n, m * n)
    expected_value = 2.0 * inputs * torch.eye(m * n)
    assert actual_value.equal(expected_value), err_str

    # Check (A1, A2) Hessian values
    if not diagonal_only:
        assert torch.all(hessian_dict["A1"]["A2"] == 0.0), err_str

    # Check (A2, A1) Hessian values
    if not diagonal_only:
        assert torch.all(hessian_dict["A2"]["A1"] == 0.0), err_str

    # Check (A2, A2) Hessian values
    actual_value = hessian_dict["A2"]["A2"].view(m * n, m * n)
    expected_value = 2.0 * inputs * torch.eye(m * n)
    assert actual_value.equal(expected_value), err_str


def test_model_hessian_dict_sum_norms_squared_frozen(
    sum_norms_squared_frozen: nn.Module,
) -> None:
    """Non-batch `model_hessian_dict()` with frozen sum-norms-squared model."""
    # Make aliases for brevity
    A1 = sum_norms_squared_frozen.A1
    A2 = sum_norms_squared_frozen.A2
    m, n = A1.shape[0], A1.shape[1]

    # Make inputs
    inputs = randint()

    # Compute Hessian dict
    hessian_dict = model_hessian_dict(
        model=sum_norms_squared_frozen,
        inputs=inputs,
        is_batched=False,
    )

    # Check keys
    err_str = "Key error"
    assert list(hessian_dict.keys()) == ["A2"], err_str
    assert list(hessian_dict["A2"].keys()) == ["A2"], err_str

    # Check Hessian shapes
    err_str = "Error in Hessian shapes"

    # Check (A2, A2) Hessian shape
    expected_shape = 2 * A2.shape
    assert hessian_dict["A2"]["A2"].shape == expected_shape, err_str

    # Check Hessian values
    err_str = "Error in Hessian values"

    # Check (A2, A2) Hessian values
    actual_value = hessian_dict["A2"]["A2"].view(m * n, m * n)
    expected_value = 2.0 * inputs * torch.eye(m * n)
    assert actual_value.equal(expected_value), err_str
