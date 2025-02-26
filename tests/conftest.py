"""Test configuration."""

# pylint: disable=invalid-name

from typing import Callable

import pytest
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker


@pytest.fixture(name="commutation_matrix")
def fixture_commutation_matrix() -> Callable[[int, int], Float[Tensor, "mn mn"]]:
    """Return a function that makes commutation matrices."""

    @torch.no_grad()
    @jaxtyped(typechecker=typechecker)
    def commutation_matrix(m: int, n: int) -> Float[Tensor, "mn mn"]:
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

    return commutation_matrix  # type: ignore


class Bilinear(nn.Module):
    """
    Bilinear model.

        (x1, x2) --> (x1)^t B x2.

    Args:
        input_dim_1: Dimension of input x1.
        input_dim_2: Dimension of input x2.
    """

    def __init__(self, input_dim_1: int, input_dim_2: int) -> None:
        super().__init__()
        self.B = nn.Bilinear(input_dim_1, input_dim_2, 1, bias=False)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Evaluate model.

        Args:
            x1: First input vector(s).
            x2: Second input vector(s).

        Returns:
            Model evaluated at `(x1, x2)`.
        """
        return self.B(x1, x2)  # type: ignore


class DoubleBilinear(nn.Module):
    """
    Double-bilinear model.

        (x1, x2) --> (x1)^t B1 B2 x2.

    Args:
        input_dim_1: First input (x1) dimension.
        inner_dim: Inner dimension.
        input_dim_2: Second input (x2) dimension.
    """

    def __init__(self, input_dim_1: int, inner_dim: int, input_dim_2: int) -> None:
        super().__init__()
        self.B1 = nn.Parameter(torch.randn(input_dim_1, inner_dim))
        self.B2 = nn.Parameter(torch.randn(inner_dim, input_dim_2))

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Evaluate model.

        Args:
            x1: First input vector(s).
            x2: Second input vector(s).

        Returns:
            Model evaluated at `(x1, x2)`.
        """
        return (x1 @ self.B1) @ (self.B2 @ x2)  # type: ignore


class SumNormsSquared(nn.Module):
    """
    Sum-norms-squared model.

        x --> x (|A1|_F^2 + |A2|_F^2), for scalar x.

    Args:
        num_rows: Number of rows in inputs A1, A2.
        num_cols: Number of columns in inputs A1, A2.
    """

    def __init__(self, num_rows: int, num_cols: int) -> None:
        super().__init__()
        self.A1 = nn.Parameter(torch.randn(num_rows, num_cols))
        self.A2 = nn.Parameter(torch.randn(num_rows, num_cols))

    def forward(self, x: Tensor) -> Tensor:
        """
        Evaluate model.

        Args:
            x: Input scalar(s).

        Returns:
            Model evaluated at `x`.
        """
        A1_norm_squared = torch.trace(self.A1.T @ self.A1)
        A2_norm_squared = torch.trace(self.A2.T @ self.A2)
        return x * (A1_norm_squared + A2_norm_squared)


@pytest.fixture(name="bilinear")
def fixture_bilinear() -> nn.Module:
    """Bilinear model."""
    return Bilinear(3, 4)


@pytest.fixture(name="double_bilinear")
def fixture_double_bilinear() -> nn.Module:
    """Double-bilinear model."""
    return DoubleBilinear(2, 3, 4)


@pytest.fixture(name="double_bilinear_frozen")
def fixture_double_bilinear_frozen() -> nn.Module:
    """Double-bilinear model with one frozen parameter."""
    model = DoubleBilinear(2, 3, 4)
    model.B1.requires_grad_(False)
    return model


@pytest.fixture(name="sum_norms_squared")
def fixture_sum_norms_squared() -> nn.Module:
    """Sum-norms-squared model."""
    return SumNormsSquared(2, 3)


@pytest.fixture(name="sum_norms_squared_frozen")
def fixture_sum_norms_squared_frozen() -> nn.Module:
    """Sum-norms-squared model with one frozen parameter."""
    model = SumNormsSquared(2, 3)
    model.A1.requires_grad_(False)
    return model
