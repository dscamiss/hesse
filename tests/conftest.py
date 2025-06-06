"""Test configuration."""

# Disable "returns Any" warnings caused by unhinted PyTorch functions
# mypy: disable-error-code="no-any-return"

# pylint: disable=invalid-name

import pytest
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from hesse.types import Criterion

_RANDINT_LO = -10
_RANDINT_HI = 10


@torch.no_grad()
@jaxtyped(typechecker=typechecker)
def randint(shape: tuple[int, ...] = ()) -> Float[Tensor, "..."]:
    """
    Return tensor with random integer values and floating-point data type.

    Args:
        shape: Tensor shape.

    Returns:
        Tensor with random integer values between `_RANDINT_LO` and
        `_RANDINT_HI`, inclusive.
    """
    return torch.randint(_RANDINT_LO, _RANDINT_HI, shape).float()


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
        return self.B(x1, x2)


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
        self.B1 = nn.Parameter(randint((input_dim_1, inner_dim)))
        self.B2 = nn.Parameter(randint((inner_dim, input_dim_2)))

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Evaluate model.

        Args:
            x1: First input vector(s).
            x2: Second input vector(s).

        Returns:
            Model evaluated at `(x1, x2)`.
        """
        return torch.sum((x1 @ self.B1) * (x2 @ self.B2.T), dim=-1)


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
        self.A1 = nn.Parameter(randint((num_rows, num_cols)))
        self.A2 = nn.Parameter(randint((num_rows, num_cols)))

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


@pytest.fixture(name="batch_size")
def fixture_batch_size() -> int:
    """Batch size for input/output data."""
    return 4


@pytest.fixture(name="mse")
def fixture_mse() -> Criterion:
    """Make MSE loss criterion (no normalization by batch size)."""
    return nn.MSELoss(reduction="sum")
