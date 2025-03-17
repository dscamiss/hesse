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


@jaxtyped(typechecker=typechecker)
def batch_hessian_matrix_from_hessian_dict(
    model: nn.Module, batch_hessian_dict: BatchHessianDict
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

    Returns:
        Batch Hessian, as a tensor.

        The output `hess` is such that `hess[b, :]` is the Hessian matrix
        corresponding to batch `b`.
    """
    # Get model parameters dict
    params_dict = dict(model.named_parameters())

    # Determine batch size
    hessian_param_names = list(batch_hessian_dict.keys())
    batch_size = params_dict[hessian_param_names[0]].shape[0]

    # Determine Hessian matrix size
    hessian_size = sum(params_dict[param_name][0, :].numel() for param_name in hessian_param_names)

    # Allocate batch Hessian matrix
    batch_hessian_matrix = torch.zeros(batch_size, hessian_size, hessian_size)

    # Populate batch Hessian matrix
    row_start = 0
    for batch in range(batch_size):
        for row_param_name in hessian_param_names:
            row_size = params_dict[row_param_name].numel()
            row_end = row_start + row_size
            col_start = 0
            col_end = 0
            for col_param_name in hessian_param_names:
                col_size = params_dict[col_param_name].numel()
                col_end = col_start + col_size
                hessian_block = batch_hessian_dict[row_param_name][col_param_name][batch, :]
                hessian_block = hessian_block.view(row_size, col_size)
                batch_hessian_matrix[batch, row_start:row_end, col_start:col_end] = hessian_block
                col_start = col_end
            row_start = row_end

    return batch_hessian_matrix


@jaxtyped(typechecker=typechecker)
def hessian_matrix_from_hessian_dict(
    model: nn.Module, hessian_dict: HessianDict
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

    # Populate Hessian matrix
    row_start = 0
    for row_param_name in hessian_param_names:
        row_size = params_dict[row_param_name].numel()
        row_end = row_start + row_size
        col_start = 0
        col_end = 0
        for col_param_name in hessian_param_names:
            col_size = params_dict[col_param_name].numel()
            col_end = col_start + col_size
            hessian_block = hessian_dict[row_param_name][col_param_name].view(row_size, col_size)
            hessian_matrix[row_start:row_end, col_start:col_end] = hessian_block
            col_start = col_end
        row_start = row_end

    return hessian_matrix


@jaxtyped(typechecker=typechecker)
def model_hessian_matrix(
    model: nn.Module, inputs: Inputs, params: Params = None
) -> Num[Tensor, "n n"]:
    """
    Hessian of a model with respect to its parameters.

    This function expects `inputs` to have no batch dimension.

    Args:
        model: Network model.
        inputs: Inputs to the model.
        params: Specific model parameters to use.  The default value is `None`
            which means use all model parameters which are not frozen.

    Returns:
        Hessian of `model` with respect to its parameters, as a matrix.
    """
    hessian_dict = model_hessian_dict(model, inputs, params)
    return hessian_matrix_from_hessian_dict(model, hessian_dict)


@jaxtyped(typechecker=typechecker)
def batch_model_hessian_matrix(
    model: nn.Module, batch_inputs: BatchInputs, params: Params = None
) -> Num[Tensor, "b n n"]:
    """
    Batch Hessian of a model with respect to its parameters.

    Args:
        model: Network model.
        batch_inputs: Batch model inputs.
        params: Specific model parameters to use.  The default value is `None`
            which means use all model parameters which are not frozen.

    Returns:
        Batch Hessian of `model` with respect to its parameters, as a tensor.

        The output `hess` is such that `hess[b, :]` is the Hessian matrix
        corresponding to batch `b`.
    """
    batch_hessian_dict = batch_model_hessian_dict(model, batch_inputs, params)
    return batch_hessian_matrix_from_hessian_dict(model, batch_hessian_dict)


@jaxtyped(typechecker=typechecker)
def loss_hessian_matrix(
    model: nn.Module,
    criterion: Criterion,
    inputs: Inputs,
    target: Target,
    params: Params = None,
) -> Num[Tensor, "n n"]:
    """
    Hessian of a loss function with respect to model parameters.

    This version expects `inputs` and `target` to have no batch dimension.

    Args:
        model: Network model.
        criterion: Loss criterion.
        inputs: Model inputs.
        target: Target model output.
        params: Specific model parameters to use.  The default value is `None`
            which means use all model parameters which are not frozen.

    Returns:
        Hessian matrix of the loss function

            `loss = criterion(model(inputs), target)`

        with respect to model parameters.
    """
    hessian_dict = loss_hessian_dict(model, criterion, inputs, target, params)
    return hessian_matrix_from_hessian_dict(model, hessian_dict)


@jaxtyped(typechecker=typechecker)
def batch_loss_hessian_matrix(
    model: nn.Module,
    criterion: Criterion,
    batch_inputs: BatchInputs,
    batch_target: BatchTarget,
    params: Params = None,
) -> Num[Tensor, "b n n"]:
    """
    Batch Hessian of a loss function with respect to model parameters.

    Args:
        model: Network model.
        criterion: Loss criterion.
        batch_inputs: Batch model inputs.
        batch_target: Batch target model output.
        params: Specific model parameters to use.  The default value is `None`
            which means use all model parameters which are not frozen.

    Returns:
        Batch Hessian of the loss function

            `loss = criterion(model(inputs), target)`

        with respect to the specified model parameters, as a tensor.

        The output `hess` is such that `hess[b, :]` is the Hessian matrix
        corresponding to batch `b`.
    """
    batch_hessian_dict = batch_loss_hessian_dict(
        model, criterion, batch_inputs, batch_target, params
    )
    return batch_hessian_matrix_from_hessian_dict(model, batch_hessian_dict)
