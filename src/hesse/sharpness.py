"""Helper functions to compute sharpness."""

# Next line disables "returns Any" errors caused by unhinted PyTorch functions
# mypy: disable-error-code="no-any-return"

import torch
from jaxtyping import Num, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from src.hesse.hessian_matrix import (
    batch_loss_hessian_matrix,
    batch_model_hessian_matrix,
    loss_hessian_matrix,
    model_hessian_matrix,
)
from src.hesse.types import BatchInputs, BatchTarget, Criterion, Inputs, Params, Target

_Scalar = Num[Tensor, ""]
_BatchScalar = Num[Tensor, "b "]


@jaxtyped(typechecker=typechecker)
def smallest_eigenvalue(matrix: Num[Tensor, "m n"]) -> _Scalar:
    """
    Smallest eigenvalue of a matrix.

    Args:
        matrix: Input matrix.

    Returns:
        Smallest eigenvalue of input matrix.
    """
    return torch.lobpcg(matrix, k=1, largest=False)[0][0]


@jaxtyped(typechecker=typechecker)
def largest_eigenvalue(matrix: Num[Tensor, "m n"]) -> _Scalar:
    """
    Largest eigenvalue of a matrix.

    Args:
        matrix: Input matrix.

    Returns:
        Largest eigenvalue of input matrix.
    """
    return torch.lobpcg(matrix, k=1, largest=True)[0][0]


@jaxtyped(typechecker=typechecker)
def sharpness(matrix: Num[Tensor, "m n"]) -> _Scalar:
    """
    Sharpness (largest absolute eigenvalue) of a matrix.

    Args:
        matrix: Input matrix.

    Returns:
        Largest absolute eigenvalue of input matrix.
    """
    abs_smallest_eigenvalue = torch.abs(smallest_eigenvalue(matrix))
    abs_largest_eigenvalue = torch.abs(largest_eigenvalue(matrix))
    return torch.max(abs_smallest_eigenvalue, abs_largest_eigenvalue)


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

    # Return its sharpness
    return sharpness(hessian_matrix)


def batch_model_sharpness(
    model: nn.Module, batch_inputs: BatchInputs, params: Params = None
) -> _BatchScalar:
    """
    Batch sharpness of a model with respect to its parameters.

    Args:
        model: Network model.
        batch_inputs: Batch model inputs.
        params: Specific model parameters to use.  The default value is `None`
            which means use all model parameters which are not frozen.

    Returns:
        Batch sharpness of `model` with respect to its parameters.

        The output `sharpness` is such that `sharpness[b]` is the sharpness
        corresponding to batch `b`.
    """
    # Make batch Hessian matrix
    batch_hessian_matrix = batch_model_hessian_matrix(model, batch_inputs, params)

    # Allocate batch sharpness
    batch_size = batch_inputs.shape[0]
    batch_sharpness = torch.zeros(batch_size)

    # Populate batch sharpness
    for batch in range(batch_size):
        batch_sharpness[batch] = sharpness(batch_hessian_matrix[batch, :])

    return batch_sharpness


def loss_sharpness(
    model: nn.Module,
    criterion: Criterion,
    inputs: Inputs,
    target: Target,
    params: Params = None,
) -> _Scalar:
    """
    Sharpness of a loss function with respect to model parameters.

    This version expects `inputs` and `target` to have no batch dimension.

    Args:
        model: Network model.
        criterion: Loss criterion.
        inputs: Model inputs.
        target: Target model output.
        params: Specific model parameters to use.  The default value is `None`
            which means use all model parameters which are not frozen.

    Returns:
        Sharpness of the loss function

            `loss = criterion(model(inputs), target)`

        with respect to model parameters.
    """
    # Make Hessian matrix
    hessian_matrix = loss_hessian_matrix(model, criterion, inputs, target, params)

    # Return its sharpness
    return sharpness(hessian_matrix)


def batch_loss_sharpness(
    model: nn.Module,
    criterion: Criterion,
    batch_inputs: BatchInputs,
    batch_target: BatchTarget,
    params: Params = None,
) -> _BatchScalar:
    """
    Batch sharpness of a loss function with respect to model parameters.

    Args:
        model: Network model.
        criterion: Loss criterion.
        batch_inputs: Batch model inputs.
        batch_target: Batch target model output.
        params: Specific model parameters to use.  The default value is `None`
            which means use all model parameters which are not frozen.

    Returns:
        Batch sharpness of the loss function

            `loss = criterion(model(inputs), target)`

        with respect to the specified model parameters.

        The output `sharpness` is such that `sharpness[b]` is the sharpness
        corresponding to batch `b`.
    """
    # Make batch Hessian matrix
    batch_hessian_matrix = batch_loss_hessian_matrix(
        model, criterion, batch_inputs, batch_target, params
    )

    # Allocate batch sharpness
    batch_size = batch_inputs.shape[0]
    batch_sharpness = torch.zeros(batch_size)

    # Populate batch sharpness
    for batch in range(batch_size):
        batch_sharpness[batch] = sharpness(batch_hessian_matrix[batch, :])

    return batch_sharpness
