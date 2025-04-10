"""Model Hessian example."""

import torch
from jaxtyping import Num, jaxtyped
from torch import Tensor
from typeguard import typechecked as typechecker

import hesse

# pylint: disable=invalid-name


class MimoModel(torch.nn.Module):
    """Multi-input, multi-output demo model."""

    def __init__(self, m: int) -> None:
        super().__init__()
        self.A = torch.nn.Parameter(torch.randn(m, m))
        self.B = torch.nn.Parameter(torch.randn(m, m))

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
    """Run model Hessian demo."""
    model = MimoModel(2)

    # Make batch inputs
    x = torch.Tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    y = -1.0 * x

    # Compute full model Hessian matrix
    model_hessian = hesse.model_hessian_matrix(model=model, inputs=(x, y))

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

    # Compare actual and expected
    assert model_hessian.equal(expected), "Error in Hessian values"
    print("Full model Hessian values match")

    # Compute reduced model Hessian matrix
    reduced_model_hessian = hesse.model_hessian_matrix(model=model, inputs=(x, y), params=("A",))

    # Compute expected reduced model Hessian matrix
    expected = torch.zeros([2, 4, 4, 4])

    expected[0][0] = 2.0 * torch.eye(4)
    expected[0][1] = 4.0 * torch.eye(4)
    expected[1][0] = 6.0 * torch.eye(4)
    expected[1][1] = 8.0 * torch.eye(4)

    # Compare actual and expected
    assert reduced_model_hessian.equal(expected), "Error in Hessian values"
    print("Reduced model Hessian values match")

    # Make loss criterion and target output
    criterion = torch.nn.MSELoss(reduction="sum")
    target = torch.randn(2, 4)

    # Compute full loss Hessian matrix
    loss_hessian = hesse.loss_hessian_matrix(
        model=model,
        criterion=criterion,
        inputs=(x, y),
        target=target,
    )

    torch.set_printoptions(linewidth=120)
    print(loss_hessian)


if __name__ == "__main__":
    run_demo()
