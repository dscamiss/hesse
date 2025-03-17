"""Sharpness functions."""

import torch
from jaxtyping import Num, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from src.hesse.hessian_matrix import model_hessian_matrix
from src.hesse.types import Inputs, Params

_Scalar = Num[Tensor, ""]
_BatchScalar = Num[Tensor, "b "]


@jaxtyped(typechecker=typechecker)
def model_sharpness(model: nn.Module, inputs: Inputs, params: Params = None) -> _Scalar:
    """
    Sharpness of a model with respect to its parameters.

    This function expects `inputs` to have no batch dimension.

    Args:
        model: Network model.
        inputs: Inputs to the model.
        params: Specific model parameters to use.  The default value is `None`
            which means use all model parameters which are not frozen.

    Returns:
        Sharpness of `model` with respect to its parameters.
    """
    # Make Hessian matrix
    hessian_matrix = model_hessian_matrix(model, inputs, params)

    # Return largest eigenvalue
    return torch.lobpcg(hessian_matrix, k=1)[0][0]


def batch_model_sharpness() -> _BatchScalar:
    """TODO."""
    return torch.as_tensor(1.0)


def loss_sharpness() -> _Scalar:
    """TODO."""
    return torch.as_tensor(1.0)


def batch_loss_sharpness() -> _BatchScalar:
    """TODO."""
    return torch.as_tensor(1.0)
