"""Test code for hesse.py."""

# pylint: disable=invalid-name

import pytest
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from src.hesse import compute_hessian


@torch.no_grad
@jaxtyped(typechecker=typechecker)
def commutation_matrix(m, n) -> Float[Tensor, "mn mn"]:
    """
    Construct the commutation matrix K_{m,n}.

    For an m-by-n input matrix A, K_{m,n} is an mn-by-mn matrix that satisfies

        K_{m,n} vec(A) = vec(A^t),

    where vec() is the column-stacking vectorization map.

    Being a permutation matrix, K_{m,n} is orthogonal and therefore

        vec(A) = K_{m,n}^t vec(A^t).

    Args:
        m: "Row dimension" argument.
        n: "Column dimension" argument.

    Returns:
        Tensor containing K_{m,n}.
    """
    indices = torch.arange(m * n).reshape(m, n).T.reshape(-1)
    return torch.eye(m * n).index_select(0, indices).T


@torch.no_grad
def test_commutation_matrix() -> None:
    """Test `commutation_matrix()`."""
    A = torch.randn(3, 4)
    K = commutation_matrix(A.shape[0], A.shape[1])

    err_str = "Error in commutation matrix"
    # Note: Transpose here since `flatten()` is *row-stacking* vectorization
    vec_A = A.T.flatten()
    vec_A_transpose = A.flatten()
    assert torch.all(K @ vec_A == vec_A_transpose), err_str


def test_compute_hessian_bilinear(bilinear: nn.Module) -> None:
    """Test `compute_hessian()` with bilinear model."""
    # Make input data
    x1 = torch.randn(bilinear.B.in1_features).requires_grad_(False)
    x2 = torch.randn(bilinear.B.in2_features).requires_grad_(False)

    # PyTorch issues performance warning for unimplemented batching rule
    # - This does not affect the correctness of the implementation.
    with pytest.warns(UserWarning):
        hess = compute_hessian(bilinear, x1, x2)

    # Check Hessian shape
    err_str = "Error in Hessian shape"
    expected_shape = 2 * bilinear.B.weight.shape
    assert hess["B.weight"]["B.weight"].shape == expected_shape, err_str

    # Check Hessian values
    err_str = "Error in Hessian values"
    assert torch.all(hess["B.weight"]["B.weight"] == 0.0), err_str


def test_compute_hessian_double_bilinear(double_bilinear: nn.Module) -> None:
    """Test `compute_hessian()` with double-bilinear model."""
    # Make aliases for brevity
    B1 = double_bilinear.B1
    B2 = double_bilinear.B2
    m, n, p = B1.shape[0], B1.shape[1], B2.shape[1]

    # Make input data
    x1 = torch.randn(m).requires_grad_(False)
    x2 = torch.randn(p).requires_grad_(False)

    # Compute Hessian
    hess = compute_hessian(double_bilinear, x1, x2)

    # Check Hessian shapes
    err_str = "Error in Hessian shape"

    # Check (B1, B1) Hessian shape
    expected_shape = 2 * B1.shape
    assert hess["B1"]["B1"].shape == expected_shape, err_str

    # Check (B1, B2) Hessian shape
    expected_shape = B1.shape + B2.shape
    assert hess["B1"]["B2"].shape == expected_shape, err_str

    # Check (B2, B1) Hessian shape
    expected_shape = B2.shape + B1.shape
    assert hess["B2"]["B1"].shape == expected_shape, err_str

    # Check (B2, B2) Hessian shape
    expected_shape = 2 * B2.shape
    assert hess["B2"]["B2"].shape == expected_shape, err_str

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
    K = commutation_matrix(m, n).requires_grad_(False)

    # Check (B1, B1) Hessian values
    assert torch.all(hess["B1"]["B1"] == 0.0), err_str

    # Check (B1, B2) Hessian values
    expected_value = K @ torch.kron(torch.eye(n), torch.outer(x1, x2))
    assert torch.allclose(hess["B1"]["B2"].view(m * n, n * p), expected_value), err_str

    # Check (B2, B1) Hessian values
    expected_value = torch.kron(torch.eye(n), torch.outer(x2, x1)) @ K.T
    assert torch.allclose(hess["B2"]["B1"].view(n * p, m * n), expected_value), err_str

    # Check (B2, B2) Hessian values
    assert torch.all(hess["B2"]["B2"] == 0.0), err_str


def test_compute_hessian_double_bilinear_frozen(double_bilinear_frozen: nn.Module) -> None:
    """Test `compute_hessian()` with frozen double-bilinear model."""
    # Make aliases for brevity
    B1 = double_bilinear_frozen.B1
    B2 = double_bilinear_frozen.B2
    m, p = B1.shape[0], B2.shape[1]

    # Make input data
    x1 = torch.randn(m).requires_grad_(False)
    x2 = torch.randn(p).requires_grad_(False)

    # Compute Hessian
    hess = compute_hessian(double_bilinear_frozen, x1, x2)

    # Check keys
    err_str = "Key error"
    assert list(hess.keys()) == ["B2"], err_str
    assert list(hess["B2"].keys()) == ["B2"], err_str

    # Check Hessian shapes
    err_str = "Error in Hessian shape"

    # Check (B2, B2) Hessian shape
    expected_shape = 2 * B2.shape
    assert hess["B2"]["B2"].shape == expected_shape, err_str

    # Check Hessian values
    err_str = "Error in Hessian values"

    # Check (B2, B2) Hessian values
    assert torch.all(hess["B2"]["B2"] == 0.0), err_str


def test_compute_hessian_sum_norms_squared(
    sum_norms_squared: nn.Module,
) -> None:
    """Test `compute_hessian()` with sum-norms-squared model."""
    # Make aliases for brevity
    A1 = sum_norms_squared.A1
    A2 = sum_norms_squared.A2
    m, n = A1.shape[0], A1.shape[1]

    # Make input data
    x = torch.randn([]).requires_grad_(False)

    # Compute Hessian
    hess = compute_hessian(sum_norms_squared, x)

    # Check Hessian shapes
    err_str = "Error in Hessian shape"

    # Check (A1, A1) Hessian shape
    expected_shape = 2 * A1.shape
    assert hess["A1"]["A1"].shape == expected_shape, err_str

    # Check (A1, A2) Hessian shape
    expected_shape = A1.shape + A2.shape
    assert hess["A1"]["A2"].shape == expected_shape, err_str

    # Check (A2, A1) Hessian shape
    expected_shape = A2.shape + A1.shape
    assert hess["A2"]["A1"].shape == expected_shape, err_str

    # Check (A2, A2) Hessian shape
    expected_shape = 2 * A2.shape
    assert hess["A2"]["A2"].shape == expected_shape, err_str

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
    expected_value = 2.0 * x * torch.eye(m * n)
    assert torch.allclose(hess["A1"]["A1"].view(m * n, m * n), expected_value), err_str

    # Check (A1, A2) Hessian values
    assert torch.all(hess["A1"]["A2"] == 0.0), err_str

    # Check (A2, A1) Hessian values
    assert torch.all(hess["A2"]["A1"] == 0.0), err_str

    # Check (A2, A2) Hessian values
    expected_value = 2.0 * x * torch.eye(m * n)
    assert torch.allclose(hess["A2"]["A2"].view(m * n, m * n), expected_value), err_str


def test_compute_hessian_sum_norms_squared_frozen(
    sum_norms_squared_frozen: nn.Module,
) -> None:
    """Test `compute_hessian()` with frozen sum-norms-squared model."""
    # Make aliases for brevity
    A1 = sum_norms_squared_frozen.A1
    A2 = sum_norms_squared_frozen.A2
    m, n = A1.shape[0], A1.shape[1]

    # Make input data
    x = torch.randn([]).requires_grad_(False)

    # Compute Hessian
    hess = compute_hessian(sum_norms_squared_frozen, x)

    # Check keys
    err_str = "Key error"
    assert list(hess.keys()) == ["A2"], err_str
    assert list(hess["A2"].keys()) == ["A2"], err_str

    # Check Hessian shapes
    err_str = "Error in Hessian shape"

    # Check (A2, A2) Hessian shape
    expected_shape = 2 * A2.shape
    assert hess["A2"]["A2"].shape == expected_shape, err_str

    # Check Hessian values
    err_str = "Error in Hessian values"

    # Check (A2, A2) Hessian values
    expected_value = 2.0 * x * torch.eye(m * n)
    assert torch.allclose(hess["A2"]["A2"].view(m * n, m * n), expected_value), err_str
