"""Toy multi-input, multi-output model."""

import torch
from jaxtyping import Num, jaxtyped
from torch import Tensor
from typeguard import typechecked as typechecker

import hesse

# pylint: disable=invalid-name


class MimoModel(torch.nn.Module):
    """Multi-input, multi-output demo model."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.A = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.B = torch.nn.Parameter(torch.randn(input_dim, output_dim))

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Num[Tensor, "b n"], y: Num[Tensor, "b n"]) -> Num[Tensor, "b two_n"]:
        """
        Run forward pass.

        Args:
            x: First input tensor of shape (b, n).
            y: Second input tensor of shape (b, n).

        Returns:
            The matrix

            [ tr(A^t A) x_{    0, :}   tr(B^t B) y_{    0, :} ]
            [ tr(A^t A) x_{    1, :}   tr(B^t B) y_{    1, :} ]
            [           :                        :            ]
            [ tr(A^t A) x_{m - 1, :}   tr(B^t B) y_{m - 1, :} ].
        """
        row_1 = torch.trace(self.A.T @ self.A) * x
        row_2 = torch.trace(self.B.T @ self.B) * y
        return torch.hstack((row_1, row_2))


def run_demo() -> None:
    """Run demo for multi-input, multi-output model."""
    input_dim = 2
    output_dim = 2
    model = MimoModel(input_dim, output_dim)

    # Make
    x = torch.Tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    y = -1.0 * x

    # Compute full Hessian matrix
    hessian = hesse.model_hessian_matrix(model=model, inputs=(x, y))

    # Compute expected Hessian matrix
    expected = torch.zeros([2, 4, 8, 8])

    expected[0][0][:4, :4] = 2.0 * torch.eye(4)
    expected[0][1][:4, :4] = 4.0 * torch.eye(4)
    expected[0][2][4:, 4:] = -2.0 * torch.eye(4)
    expected[0][3][4:, 4:] = -4.0 * torch.eye(4)
    expected[1][0][:4, :4] = 6.0 * torch.eye(4)
    expected[1][1][:4, :4] = 8.0 * torch.eye(4)
    expected[1][2][4:, 4:] = -6.0 * torch.eye(4)
    expected[1][3][4:, 4:] = -8.0 * torch.eye(4)

    assert hessian.equal(expected), "Error in Hessian values"

    print("Full Hessian values match")

    # Compute reduced Hessian matrix
    hessian = hesse.model_hessian_matrix(model=model, inputs=(x, y), params=("A"))

    # Compute expected reduced Hessian matrix
    expected = torch.zeros([2, 4, 4, 4])

    expected[0][0] = 2.0 * torch.eye(4)
    expected[0][1] = 4.0 * torch.eye(4)
    expected[1][0] = 6.0 * torch.eye(4)
    expected[1][1] = 8.0 * torch.eye(4)

    assert hessian.equal(expected), "Error in Hessian values"

    print("Reduced Hessian values match")


if __name__ == "__main__":
    run_demo()
