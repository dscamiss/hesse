"""Test configuration."""

# pylint: disable=invalid-name

import pytest
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker


class Bilinear(nn.Module):
    """
    Bilinear model.

        (x1, x2) --> (x1)^t B x2.

    Args:
        input_dim_1: First input dimension (x1).
        input_dim_2: Second input dimension (x2).
    """

    def __init__(self, input_dim_1: int, input_dim_2: int) -> None:
        super().__init__()
        self.B = nn.Bilinear(input_dim_1, input_dim_2, 1, bias=False)

    @jaxtyped(typechecker=typechecker)
    def forward(self, x1: Float[Tensor, " n1"], x2: Float[Tensor, " n2"]) -> Float[Tensor, ""]:
        """
        Evaluate model.

        Args:
            x1: First input.
            x2: Second input.

        Returns:
            Model evaluated at `(x1, x2)`.
        """
        return self.B(x1, x2).squeeze()


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

    @jaxtyped(typechecker=typechecker)
    def forward(self, x1: Float[Tensor, " n1"], x2: Float[Tensor, " n2"]) -> Float[Tensor, ""]:
        """
        Evaluate model.

        Args:
            x1: First input.
            x2: Second input.

        Returns:
            Model evaluated at `(x1, x2)`.
        """
        return (x1 @ self.B1) @ (self.B2 @ x2)


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
