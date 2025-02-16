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
    """Test `commutation_matrix()` behavior."""
    A = torch.randn(3, 4)
    K = commutation_matrix(A.shape[0], A.shape[1])

    err_str = "Error in commutation matrix"
    # Note: Transpose here since `flatten()` is *row-stacking* vectorization
    vec_A = A.T.flatten()
    vec_A_transpose = A.flatten()
    assert torch.all(K @ vec_A == vec_A_transpose), err_str


def test_compute_hessian_bilinear(bilinear: nn.Module) -> None:
    """Test `compute_hessian()` behavior with bilinear model."""
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
    """Test `compute_hessian()` behavior with double-bilinear model."""
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
    #   * K is the commutation matrix K = K_{n,p}.
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


@jaxtyped(typechecker=typechecker)
def test_compute_hessian_double_bilinear_frozen(double_bilinear_frozen: nn.Module) -> None:
    """Test `compute_hessian()` behavior with double-bilinear model."""
    # Make aliases for brevity
    B1 = double_bilinear_frozen.B1
    B2 = double_bilinear_frozen.B2
    m, n, p = B1.shape[0], B1.shape[1], B2.shape[1]

    # Make input data
    x1 = torch.randn(m).requires_grad_(False)
    x2 = torch.randn(p).requires_grad_(False)

    # Compute Hessian
    hess = compute_hessian(double_bilinear_frozen, x1, x2)

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
