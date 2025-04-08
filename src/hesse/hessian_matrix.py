"""Functions to help compute Hessian matrices."""

# Next line disables "returns Any" errors caused by unhinted PyTorch functions
# mypy: disable-error-code="no-any-return"

from typing import Union

import torch
from jaxtyping import Num, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from src.hesse.hessian_dict import loss_hessian_dict, model_hessian_dict
from src.hesse.types import (
    BatchHessianDict,
    BatchHessianMatrix,
    BatchInputs,
    BatchTarget,
    Criterion,
    HessianDict,
    HessianMatrix,
    Inputs,
    Params,
    Target,
)


# TODO: Can we refactor to use `vmap` here?
@jaxtyped(typechecker=typechecker)
def hessian_matrix_from_hessian_dict(
    model: nn.Module,
    hessian_dict: Union[HessianDict, BatchHessianDict],
    diagonal_only: bool,
    is_batch: bool = True,
) -> Union[HessianMatrix, BatchHessianMatrix]:
    """
    Hessian (or batch Hessian) matrix from Hessian (or batch Hessian) dict.

    The ordering of the Hessian matrix blocks follows the ordering of the keys
    in `hessian_dict`.

    Args:
        model: Network model.
        hessian_dict: Hessian (or batch Hessian) dict.
        diagonal_only: Make diagonal blocks only.
        is_batch: Batch Hessian dict provided.  Default value is `True`.

    Returns:
        Hessian (or batch Hessian) matrix.

        When `hessian_dict` is batch, the output `hessian_matrix` is such
        that `hessian_matrix[b, :]` is the Hessian matrix corresponding to
        batch `b`.

        When `hessian_dict` is non-batch, the output `hessian_matrix` is
        such that `hessian_dict` is the Hessian matrix.

        If `diagonal_only` is `True`, then the non-diagonal blocks of
        `hessian_dict` (or each `hessian_dict[b, :]`) are all zeroes.
    """
    # Get model parameters dict
    params_dict = dict(model.named_parameters())

    # Determine batch size
    hessian_param_names = list(hessian_dict.keys())
    if is_batch:
        param_name = hessian_param_names[0]
        batch_size = hessian_dict[param_name][param_name].shape[0]
    else:
        # Non-batch case, uses fake batch size of 1 for code commonality
        batch_size = 1

    # Determine Hessian matrix size
    hessian_size = sum(params_dict[param_name].numel() for param_name in hessian_param_names)

    # Allocate Hessian or (batch Hessian) matrix
    hessian_matrix = torch.zeros(batch_size, hessian_size, hessian_size)

    # Populate batch Hessian matrix -- diagonal blocks
    for batch in range(batch_size):
        offset = 0
        for param_name in hessian_param_names:
            param_size = params_dict[param_name].numel()
            if is_batch:
                hessian_block = hessian_dict[param_name][param_name][batch, :]
            else:
                hessian_block = hessian_dict[param_name][param_name]
            hessian_block = hessian_block.view(param_size, param_size)
            index_slice = slice(offset, offset + param_size)
            hessian_matrix[batch, index_slice, index_slice] = hessian_block
            offset += param_size

    # If `diagonal_only` is `True`, there's no more work to do
    if diagonal_only:
        # Non-batch case, remove fake batch size
        if not is_batch:
            hessian_matrix.squeeze_(0)
        return hessian_matrix

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
                if is_batch:
                    hessian_block = hessian_dict[row_param_name][col_param_name][batch, :]
                else:
                    hessian_block = hessian_dict[row_param_name][col_param_name]
                hessian_block = hessian_block.view(row_param_size, col_param_size)
                row_slice = slice(row_offset, row_offset + row_param_size)
                col_slice = slice(col_offset, col_offset + col_param_size)
                hessian_matrix[batch, row_slice, col_slice] = hessian_block
                col_offset += col_param_size
            row_offset += row_param_size

    # Non-batch case, remove fake batch size
    if not is_batch:
        hessian_matrix.squeeze_(0)

    return hessian_matrix


@jaxtyped(typechecker=typechecker)
def model_hessian_matrix(
    model: nn.Module,
    inputs: Union[Inputs, BatchInputs],
    params: Params = None,
    diagonal_only: bool = False,
    is_batch: bool = True,
) -> Union[HessianMatrix, BatchHessianMatrix]:
    """
    Hessian (or batch Hessian) of a model with respect to its parameters.

    Args:
        model: Network model.
        inputs: Inputs (or batch inputs) to the model.
        params: Specific model parameters to use.  Default value is `None`,
            which means use all model parameters which are not frozen.
        diagonal_only: Make diagonal blocks only.  Default value is `False`.
        is_batch: Batch inputs provided.  Default value is `True`.

    Returns:
        Hessian (or batch Hessian) matrix of `model` with respect to its
        parameters.
    """
    hessian_dict = model_hessian_dict(
        model=model,
        inputs=inputs,
        params=params,
        diagonal_only=diagonal_only,
        is_batch=is_batch,
    )

    # TODO: Add number of dimensions check

    return hessian_matrix_from_hessian_dict(
        model=model,
        hessian_dict=hessian_dict,
        diagonal_only=diagonal_only,
        is_batch=is_batch,
    )


@jaxtyped(typechecker=typechecker)
def loss_hessian_matrix(
    model: nn.Module,
    criterion: Criterion,
    inputs: Union[Inputs, BatchInputs],
    target: Union[Target, BatchTarget],
    params: Params = None,
    diagonal_only: bool = False,
    is_batch: bool = True,
) -> Num[Tensor, "n n"]:
    """
    Hessian (or batch Hessian) of loss with respect to model parameters.

    Args:
        model: Network model.
        criterion: Loss criterion.
        inputs: Inputs (or batch inputs) to the model.
        target: Target output (or batch output) from the model.
        params: Specific model parameters to use.  Default value is `None`,
            which means use all model parameters which are not frozen.
        diagonal_only: Make diagonal blocks only.  Default value is `False`.
        is_batch: Batch inputs and target provided.  Default value is 'True'.

    Returns:
        Hessian (or batch Hessian) matrix of the loss function

            `loss = criterion(model(inputs), target)`

        with respect to the parameters of `model`.
    """
    hessian_dict = loss_hessian_dict(
        model=model,
        criterion=criterion,
        inputs=inputs,
        target=target,
        params=params,
        diagonal_only=diagonal_only,
    )

    # TODO: Add number of dimensions check

    return hessian_matrix_from_hessian_dict(
        model=model,
        hessian_dict=hessian_dict,
        diagonal_only=diagonal_only,
        is_batch=is_batch,
    )
