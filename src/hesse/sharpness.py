"""Functions to compute sharpness."""

# Next line disables "returns Any" errors caused by unhinted PyTorch functions
# mypy: disable-error-code="no-any-return"

from typing import Union

import torch
from jaxtyping import Num, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from hesse.hessian_matrix import loss_hessian_matrix, model_hessian_matrix
from hesse.types import BatchInputs, BatchTarget, Criterion, Inputs, Params, Target

_Scalar = Num[Tensor, "..."]
_BatchScalar = Num[Tensor, "b ..."]

_Matrix = Num[Tensor, "... m n"]
_BatchMatrix = Num[Tensor, "b ... m n"]


@jaxtyped(typechecker=typechecker)
def _smallest_eigenvalue(matrix: _Matrix) -> _Scalar:
    """
    Smallest eigenvalue of a matrix.

    Args:
        matrix: Input matrix.

    Returns:
        Smallest eigenvalue of input matrix.
    """
    return torch.lobpcg(matrix, k=1, largest=False)[0][0]


@jaxtyped(typechecker=typechecker)
def _largest_eigenvalue(matrix: _Matrix) -> _Scalar:
    """
    Largest eigenvalue of a matrix.

    Args:
        matrix: Input matrix.

    Returns:
        Largest eigenvalue of input matrix.
    """
    return torch.lobpcg(matrix, k=1, largest=True)[0][0]


@jaxtyped(typechecker=typechecker)
def _sharpness(matrix: _Matrix) -> _Scalar:
    """
    Sharpness (largest absolute eigenvalue) of a matrix.

    Args:
        matrix: Input matrix.

    Returns:
        Largest absolute eigenvalue of input matrix.
    """
    abs_smallest_eigenvalue = torch.abs(_smallest_eigenvalue(matrix))
    abs_largest_eigenvalue = torch.abs(_largest_eigenvalue(matrix))
    return torch.max(abs_smallest_eigenvalue, abs_largest_eigenvalue)


@jaxtyped(typechecker=typechecker)
def sharpness(
    matrix: Union[_Matrix, _BatchMatrix], is_batch: bool = True
) -> Union[_Scalar, _BatchScalar]:
    """
    Sharpness (or batch sharpness) of a matrix (or batch matrix).

    Args:
        matrix: Input (or batch input) matrix.
        is_batch: Batch input matrix provided.  Default value is `True`.

    Returns:
        Sharpness (or batch sharpness) of `matrix`.

    Raises:
        RuntimeError: If `matrix` has an unsupported output shape.
    """
    batch_size = matrix.shape_metadata["batch_size"]
    output_shape = matrix.shape_metadata["output_shape"]
    output_size = matrix.shape_metadata["output_size"]
    zero_dim_output = matrix.shape_metadata["zero_dim_output"]

    # TODO: Handle high-dimensional output shapes
    if len(output_shape) >= 2:
        raise RuntimeError("Unsupported high-dimensional output shape")

    # Allocate sharpness
    matrix_sharpness = torch.zeros(batch_size, output_size)

    # Populate sharpness
    for batch in range(batch_size):
        for output in range(output_size):
            index_tuple = (batch,) if is_batch else ()
            if not zero_dim_output:
                index_tuple += (output,)
            matrix_sharpness[batch, output] = _sharpness(matrix[index_tuple])

    # Postprocess to remove fictitious dimensions
    if not is_batch:
        matrix_sharpness.squeeze_(0)
        if zero_dim_output:
            matrix_sharpness.squeeze_(0)
    else:
        if zero_dim_output:
            matrix_sharpness.squeeze_(1)

    return matrix_sharpness


@jaxtyped(typechecker=typechecker)
def model_sharpness(
    model: nn.Module,
    inputs: Union[Inputs, BatchInputs],
    params: Params = None,
    is_batch: bool = True,
) -> Union[_Scalar, _BatchScalar]:
    """
    Sharpness (or batch sharpness) of a model with respect to its parameters.

    Args:
        model: Network model.
        inputs: Inputs (or batch inputs) to the model.
        params: Specific model parameters to use.  Default value is `None`,
            which means use all model parameters which are not frozen.
        is_batch: Batch inputs provided.  Default value is `True`.

    Returns:
        Sharpness (or batch sharpness) of `model` with respect to its
        parameters.
    """
    hessian_matrix = model_hessian_matrix(
        model=model,
        inputs=inputs,
        params=params,
        is_batch=is_batch,
    )

    return sharpness(hessian_matrix, is_batch)


@jaxtyped(typechecker=typechecker)
def loss_sharpness(
    model: nn.Module,
    criterion: Criterion,
    inputs: Union[Inputs, BatchInputs],
    target: Union[Target, BatchTarget],
    params: Params = None,
    is_batch: bool = True,
) -> Union[_Scalar, _BatchScalar]:
    """
    Sharpness (or batch sharpness) of loss with respect to model parameters.

    Args:
        model: Network model.
        criterion: Loss criterion.
        inputs: Inputs (or batch inputs) to the model.
        target: Target output (or batch target output) from the model.
        params: Specific model parameters to use.  Default value is `None`,
            which means use all model parameters which are not frozen.
        is_batch: Batch inputs and batch target provided.  Default value is
            `True`.

    Returns:
        Sharpness (or batch sharpness) of the loss function

            `loss = criterion(model(inputs), target)`

        with respect to the parameters of `model`.
    """
    hessian_matrix = loss_hessian_matrix(
        model=model,
        criterion=criterion,
        inputs=inputs,
        target=target,
        params=params,
        is_batch=is_batch,
    )

    return sharpness(hessian_matrix, is_batch)
