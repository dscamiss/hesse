"""Test code for non-batch `loss_hessian_dict()`."""

# pylint: disable=invalid-name,too-many-statements

import pytest
import torch
from torch import nn

from hesse import loss_hessian_dict
from hesse.types import Criterion
from tests.conftest import commutation_matrix, randint


def test_loss_hessian_dict_bilinear(bilinear: nn.Module, mse: Criterion) -> None:
    """Non-batch `loss_hessian_dict()` with bilinear model."""
    # Make inputs
    m, n = bilinear.B.in1_features, bilinear.B.in2_features
    x1 = randint((m,))
    x2 = randint((n,))
    inputs = (x1, x2)
    target = randint()

    # Compute Hessian dict
    # - PyTorch issues performance warning for unimplemented batching rule
    # - This does not affect the correctness of the implementation.
    with pytest.warns(UserWarning):
        hessian_dict = loss_hessian_dict(
            model=bilinear,
            criterion=mse,
            inputs=inputs,
            target=target,
        )

    # Check keys
    err_str = "Key error"
    assert list(hessian_dict.keys()) == ["B.weight"], err_str
    assert list(hessian_dict["B.weight"].keys()) == ["B.weight"], err_str

    # Check Hessian shapes
    err_str = "Error in Hessian shapes"
    expected_shape = 2 * bilinear.B.weight.shape
    assert hessian_dict["B.weight"]["B.weight"].shape == expected_shape, err_str

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
    K = commutation_matrix(m, n)

    flat_1 = torch.outer(x1, x2).flatten()
    flat_2 = torch.outer(x2, x1).flatten()
    actual_value = hessian_dict["B.weight"]["B.weight"].view(m * n, m * n)
    expected_value = 2.0 * (flat_1.unsqueeze(-1) @ flat_2.unsqueeze(-1).T @ K.T)
    assert actual_value.equal(expected_value), err_str


@pytest.mark.parametrize("diagonal_only", [True, False])
def test_loss_hessian_dict_double_bilinear(
    double_bilinear: nn.Module, mse: Criterion, diagonal_only: bool
) -> None:
    """Non-batch `loss_hessian_dict()` with double-bilinear model."""
    # Make aliases for brevity
    B1 = double_bilinear.B1
    B2 = double_bilinear.B2
    m, n, p = B1.shape[0], B1.shape[1], B2.shape[1]

    # Make inputs
    x1 = randint((m,))
    x2 = randint((p,))
    inputs = (x1, x2)
    target = randint()

    # Compute Hessian dict
    hessian_dict = loss_hessian_dict(
        model=double_bilinear,
        criterion=mse,
        inputs=inputs,
        target=target,
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
    # - Define
    #
    #       g(B1, B2) = (f(B1, B2) - target)^2 = e(B1, B2)^2.
    #
    # - Note that
    #
    #       dg(B1, B2) . (V1, V2) = 2 e(B1, B2) df(B1, B2) . (V1, V2)
    #
    #   and
    #
    #       d^2 g(B1, B2) . [(V1, V2), (W1, W2)]
    #           = 2 df(B1, B2) . (V1, V2) df(B1, B2) . (W1, W2)
    #               + 2 e(B1, B2) d^2 f(B1, B2) . [(V1, V2), (W1, W2)].    (*)
    #
    # - The first term of (*) is
    #
    #       2 (x^t W1 B2 y + x^t B1 W2 y) (x^t V1 B2 y + x^t B1 V2 y)
    #           = 2 x^t W1 B2 y x^t V1 B2 y + 2 x^t W1 B2 y x^t B1 V2 y
    #               + 2 x^t B1 W2 y x^t V1 B2 y + 2 x^t B1 W2 y x^t B1 V2 y.
    #
    #   To evaluate this, observe that
    #
    #       2 x^t W1 B2 y x^t V1 B2 y
    #           = 2 tr(x^t V1 B2 y) tr(x^t W1 B2 y)
    #           = 2 tr(V1 B2 y x^t) tr(B2 y x^t W1)
    #           = 2 flat(V1)^t flat(x y^t B2^t) flat(B2 y x^t)^t K^t flat(W1)
    #
    #   and similarly
    #
    #       2 x^t B1 W2 y x^t V1 B2 y
    #           = 2 tr(x^t V1 B2 y) tr(x^t B1 W2 y)
    #           = 2 tr(V1 B2 y x^t) tr(y x^t B1 W2)
    #           = 2 flat(V1)^t flat(x y^t B2^t) flat(y x^t B1)^t K^t flat(W2)
    #
    #       2 x^t W1 B2 y x^t B1 V2 y
    #           = 2 tr(x^t B1 V2 y) tr(x^t W1 B2 y)
    #           = 2 tr(V2 y x^t B1) tr(B2 y x^t W1)
    #           = 2 flat(V2)^t flat(B1^t x y^t) flat(B2 y x^t)^t K^t flat(W1)
    #
    #       2 x^t B1 W2 y x^t B1 V2 y
    #           = 2 tr(x^t B1 V2 y) tr(x^t B1 W2 y)
    #           = 2 tr(V2 y x^t B1) tr(y x^t B1 W2)
    #           = 2 flat(V2)^t flat(B1^t x y^t) flat(y x^t B1)^t K^t flat(W2).
    #
    # - The second term of (*) is
    #
    #       2 e(B1, B2) d^2 f(B1, B2) . [(V1, V2), (W1, W2)]
    #           = 2 e(B1, B2) flat(V1)^t K (I (X) x y^t) flat(W2)
    #               + 2 e(B1, B2) flat(V2)^t (I (X) y x^t) K^t flat(W1).
    #
    #   See comments in `test_model_hessian_dict.py` for derivations.
    #
    # - This means that the Hessian blocks are
    #      Hess_{B1,B1} = 2 flat(x y^t B2^t) flat(B2 y x^t)^t K^t
    #      Hess_{B1,B2} = 2 flat(x y^t B2^t) flat(y x^t B1)^t K^t
    #                       + 2 e(B1, B2) K (I (X) x y^t)
    #      Hess_{B2,B1} = 2 flat(B1^t x y^t) flat(B2 y x^t)^t K^t
    #                       + 2 e(B1, B2) (I (X) y x^t) K^t
    #      Hess_{B2,B2} = 2 flat(B1 x y^t) flat(y x^t B1)^t K^t.

    err_str = "Error in Hessian values"
    K_mn = commutation_matrix(m, n)
    K_np = commutation_matrix(n, p)
    outer_prod = torch.outer(x1, x2)
    err = double_bilinear(x1, x2) - target

    # Check (B1, B1) Hessian values
    prod = outer_prod @ B2.T
    prod_flat = prod.flatten().unsqueeze(-1)
    prod_transpose_flat = prod.T.flatten().unsqueeze(0)  # Implicit transpose
    actual_value = hessian_dict["B1"]["B1"].view(m * n, m * n)
    expected_value = 2.0 * prod_flat @ prod_transpose_flat @ K_mn.T
    assert actual_value.equal(expected_value), err_str

    # Check (B1, B2) Hessian values
    if not diagonal_only:
        prod_left = outer_prod @ B2.T
        prod_right = outer_prod.T @ B1
        prod_left_flat = prod_left.flatten().unsqueeze(-1)
        prod_right_flat = prod_right.flatten().unsqueeze(0)  # Implicit transpose
        actual_value = hessian_dict["B1"]["B2"].view(m * n, n * p)
        expected_value_1 = 2.0 * prod_left_flat @ prod_right_flat @ K_np.T
        expected_value_2 = 2.0 * err * K_mn @ torch.kron(torch.eye(n), outer_prod)
        expected_value = expected_value_1 + expected_value_2
        assert actual_value.equal(expected_value), err_str

    # Check (B2, B1) Hessian values
    if not diagonal_only:
        prod_left = B1.T @ outer_prod
        prod_right = B2 @ outer_prod.T
        prod_left_flat = prod_left.flatten().unsqueeze(-1)
        prod_right_flat = prod_right.flatten().unsqueeze(0)  # Implicit transpose
        actual_value = hessian_dict["B2"]["B1"].view(n * p, m * n)
        expected_value_1 = 2.0 * prod_left_flat @ prod_right_flat @ K_mn.T
        # Recomputing the outer product is necessary here due to bugs:
        # - https://github.com/pytorch/pytorch/issues/54135
        # - https://github.com/pytorch/pytorch/issues/74442
        outer_prod_transpose = torch.outer(x2, x1)
        expected_value_2 = 2.0 * err * torch.kron(torch.eye(n), outer_prod_transpose) @ K_mn.T
        expected_value = expected_value_1 + expected_value_2
        assert actual_value.equal(expected_value), err_str

    # Check (B2, B2) Hessian values
    prod = B1.T @ outer_prod
    prod_flat = prod.flatten().unsqueeze(-1)
    prod_transpose_flat = prod.T.flatten().unsqueeze(0)  # Implicit transpose
    actual_value = hessian_dict["B2"]["B2"].view(n * p, n * p)
    expected_value = 2.0 * prod_flat @ prod_transpose_flat @ K_np.T
    assert actual_value.equal(expected_value), err_str
