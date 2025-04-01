"""Helper functions to compute Hessian matrices."""

# Next line disables "returns Any" errors caused by unhinted PyTorch functions
# mypy: disable-error-code="no-any-return"

import torch
from jaxtyping import Num, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from src.hesse.hessian_dict import (
    batch_loss_hessian_dict,
    batch_model_hessian_dict,
    loss_hessian_dict,
    model_hessian_dict,
)
from src.hesse.types import (
    BatchHessianDict,
    BatchInputs,
    BatchTarget,
    Criterion,
    HessianDict,
    Inputs,
    Params,
    Target,
)


# TODO: Can we use sparse storage for `diagonal_only` case?
@jaxtyped(typechecker=typechecker)
def hessian_matrix_from_hessian_dict(
    model: nn.Module, hessian_dict: HessianDict, diagonal_only: bool
) -> Num[Tensor, "n n"]:
    """
    Make Hessian matrix from Hessian represented as a dict.

    The ordering of the matrix blocks follows the ordering of the keys in
    `hessian_dict`.  We do not require a particular block ordering -- so far,
    we are only interested in its eigenvalues, which are invariant to the
    block ordering.

    Args:
        model: Model.
        hessian_dict: Hessian dict.
        diagonal_only: Make diagonal data only.

    Returns:
        Hessian matrix.
    """
    # Get model parameters dict
    params_dict = dict(model.named_parameters())

    # Determine Hessian matrix size
    hessian_param_names = list(hessian_dict.keys())
    hessian_size = sum(params_dict[param_name].numel() for param_name in hessian_param_names)

    # Allocate Hessian matrix
    hessian_matrix = torch.zeros(hessian_size, hessian_size)

    # Populate Hessian matrix -- diagonal blocks
    offset = 0
    for param_name in hessian_param_names:
        param_size = params_dict[param_name].numel()
        hessian_block = hessian_dict[param_name][param_name][:]
        hessian_block = hessian_block.view(param_size, param_size)
        index_slice = slice(offset, offset + param_size)
        hessian_matrix[index_slice, index_slice] = hessian_block
        offset += param_size

    # If `diagonal_only` is `True`, there's no more work to do
    if diagonal_only:
        return hessian_matrix

    # Populate Hessian matrix -- off-diagonal blocks
    row_offset = 0
    for row_param_name in hessian_param_names:
        row_param_size = params_dict[row_param_name].numel()
        col_offset = 0
        for col_param_name in hessian_param_names:
            col_param_size = params_dict[col_param_name].numel()
            # Skip diagonal blocks
            if row_param_name == col_param_name:
                col_offset += col_param_size
                continue
            hessian_block = hessian_dict[row_param_name][col_param_name]
            hessian_block = hessian_block.view(row_param_size, col_param_size)
            row_slice = slice(row_offset, row_offset + row_param_size)
            col_slice = slice(col_offset, col_offset + col_param_size)
            hessian_matrix[row_slice, col_slice] = hessian_block
            col_offset += col_param_size
        row_offset += row_param_size

    return hessian_matrix


# TODO: Can we refactor to use `vmap` here?
@jaxtyped(typechecker=typechecker)
def batch_hessian_matrix_from_hessian_dict(
    model: nn.Module, batch_hessian_dict: BatchHessianDict, diagonal_only: bool
) -> Num[Tensor, "b n n"]:
    """
    Make batch Hessian matrix from batch Hessian represented as a dict.

    The ordering of the matrix blocks follows the ordering of the keys in
    `batch_hessian_dict`.  We do not require a particular block ordering --
    so far, we are only interested in its eigenvalues, which are invariant to
    the block ordering.

    Args:
        model: Model.
        batch_hessian_dict: Batch Hessian dict.
        diagonal_only: Make diagonal data only.

    Returns:
        Batch Hessian, as a tensor.

        The output `hess` is such that `hess[b, :]` is the Hessian matrix
        corresponding to batch `b`.
    """
    # Get model parameters dict
    params_dict = dict(model.named_parameters())

    # Determine batch size
    hessian_param_names = list(batch_hessian_dict.keys())
    param_name = hessian_param_names[0]
    batch_size = batch_hessian_dict[param_name][param_name].shape[0]

    # Determine Hessian matrix size
    hessian_size = sum(params_dict[param_name].numel() for param_name in hessian_param_names)

    # Allocate batch Hessian matrix
    batch_hessian_matrix = torch.zeros(batch_size, hessian_size, hessian_size)

    # Populate batch Hessian matrix -- diagonal blocks
    for batch in range(batch_size):
        offset = 0
        for param_name in hessian_param_names:
            param_size = params_dict[param_name].numel()
            hessian_block = batch_hessian_dict[param_name][param_name][batch, :]
            hessian_block = hessian_block.view(param_size, param_size)
            index_slice = slice(offset, offset + param_size)
            batch_hessian_matrix[batch, index_slice, index_slice] = hessian_block
            offset += param_size

    # If `diagonal_only` is `True`, there's no more work to do
    if diagonal_only:
        return batch_hessian_matrix

    # Populate batch Hessian matrix -- off-diagonal blocks
    for batch in range(batch_size):
        row_offset = 0
        for row_param_name in hessian_param_names:
            row_param_size = params_dict[row_param_name].numel()
            col_offset = 0
            for col_param_name in hessian_param_names:
                col_param_size = params_dict[col_param_name].numel()
                # Skip diagonal blocks
                if row_param_name == col_param_name:
                    col_offset += col_param_size
                    continue
                hessian_block = batch_hessian_dict[row_param_name][col_param_name][batch, :]
                hessian_block = hessian_block.view(row_param_size, col_param_size)
                row_slice = slice(row_offset, row_offset + row_param_size)
                col_slice = slice(col_offset, col_offset + col_param_size)
                batch_hessian_matrix[batch, row_slice, col_slice] = hessian_block
                col_offset += col_param_size
            row_offset += row_param_size

    return batch_hessian_matrix


@jaxtyped(typechecker=typechecker)
def model_hessian_matrix(
    model: nn.Module, inputs: Inputs, params: Params = None, diagonal_only: bool = False
) -> Num[Tensor, "n n"]:
    """
    Hessian of a model with respect to its parameters.

    This function expects `inputs` to have no batch dimension.

    Args:
        model: Network model.
        inputs: Inputs to the model.
        params: Specific model parameters to use.  Default value is `None`,
            which means use all model parameters which are not frozen.
        diagonal_only: Make diagonal data only.  Default value is `False`.

    Returns:
        Hessian of `model` with respect to its parameters, as a matrix.
    """
    hessian_dict = model_hessian_dict(
        model=model,
        inputs=inputs,
        params=params,
        diagonal_only=diagonal_only,
    )

    return hessian_matrix_from_hessian_dict(model, hessian_dict, diagonal_only)


@jaxtyped(typechecker=typechecker)
def batch_model_hessian_matrix(
    model: nn.Module, batch_inputs: BatchInputs, params: Params = None, diagonal_only: bool = False
) -> Num[Tensor, "b n n"]:
    """
    Batch Hessian of a model with respect to its parameters.

    Args:
        model: Network model.
        batch_inputs: Batch model inputs.
        params: Specific model parameters to use.  Default value is `None`,
            which means use all model parameters which are not frozen.
        diagonal_only: Make diagonal data only.  Default value is `False`.

    Returns:
        Batch Hessian of `model` with respect to its parameters, as a tensor.

        The output `hess` is such that `hess[b, :]` is the Hessian matrix
        corresponding to batch `b`.
    """
    batch_hessian_dict = batch_model_hessian_dict(
        model=model,
        batch_inputs=batch_inputs,
        params=params,
        diagonal_only=diagonal_only,
    )

    return batch_hessian_matrix_from_hessian_dict(model, batch_hessian_dict, diagonal_only)


@jaxtyped(typechecker=typechecker)
def loss_hessian_matrix(
    model: nn.Module,
    criterion: Criterion,
    inputs: Inputs,
    target: Target,
    params: Params = None,
    diagonal_only: bool = False,
) -> Num[Tensor, "n n"]:
    """
    Hessian of a loss function with respect to model parameters.

    This version expects `inputs` and `target` to have no batch dimension.

    Args:
        model: Network model.
        criterion: Loss criterion.
        inputs: Model inputs.
        target: Target model output.
        params: Specific model parameters to use.  Default value is `None`,
            which means use all model parameters which are not frozen.
        diagonal_only: Make diagonal data only.  Default value is `False`.

    Returns:
        Hessian matrix of the loss function

            `loss = criterion(model(inputs), target)`

        with respect to model parameters.
    """
    hessian_dict = loss_hessian_dict(
        model=model,
        criterion=criterion,
        inputs=inputs,
        target=target,
        params=params,
        diagonal_only=diagonal_only,
    )
    return hessian_matrix_from_hessian_dict(model, hessian_dict, diagonal_only)


@jaxtyped(typechecker=typechecker)
def batch_loss_hessian_matrix(
    model: nn.Module,
    criterion: Criterion,
    batch_inputs: BatchInputs,
    batch_target: BatchTarget,
    params: Params = None,
    diagonal_only: bool = False,
) -> Num[Tensor, "b n n"]:
    """
    Batch Hessian of a loss function with respect to model parameters.

    Args:
        model: Network model.
        criterion: Loss criterion.
        batch_inputs: Batch model inputs.
        batch_target: Batch target model output.
        params: Specific model parameters to use.  Default value is `None`,
            which means use all model parameters which are not frozen.
        diagonal_only: Make diagonal data only.  Default value is `False`.

    Returns:
        Batch Hessian of the loss function

            `loss = criterion(model(inputs), target)`

        with respect to the specified model parameters, as a tensor.

        The output `hess` is such that `hess[b, :]` is the Hessian matrix
        corresponding to batch `b`.
    """
    batch_hessian_dict = batch_loss_hessian_dict(
        model=model,
        criterion=criterion,
        batch_inputs=batch_inputs,
        batch_target=batch_target,
        params=params,
        diagonal_only=diagonal_only,
    )

    return batch_hessian_matrix_from_hessian_dict(model, batch_hessian_dict, diagonal_only)
