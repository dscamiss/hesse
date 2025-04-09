"""Functions to compute Hessian matrices."""

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
    BatchInputs,
    BatchTarget,
    Criterion,
    HessianDict,
    Inputs,
    Params,
    Target,
)

_HessianMatrix = Num[Tensor, "... n n"]
_BatchHessianMatrix = Num[Tensor, "b ... n n"]


def _get_non_batched_output_shape(
    model: nn.Module, inputs: Union[Inputs, BatchInputs], is_batch: bool
) -> torch.Size:
    """
    Get the non-batched output shape of a model.

    Args:
        model: Network model.
        inputs: Inputs (or batch inputs) to the model.
        is_batch: Batch inputs provided.

    Returns:
        Non-batched model output shape.
    """
    # Put model into eval mode to avoid side effects, ensure repeatability
    save_training_mode = model.training
    model.eval()

    output_shape = model(*inputs).shape
    if is_batch:
        output_shape = output_shape[1:]

    # Restore training mode
    model.train(save_training_mode)

    return output_shape


# TODO: Can we refactor to use `vmap` here?
@jaxtyped(typechecker=typechecker)
def hessian_matrix_from_hessian_dict(
    model: nn.Module,
    inputs: Union[Inputs, BatchInputs],
    hessian_dict: Union[HessianDict, BatchHessianDict],
    diagonal_only: bool,
    is_batch: bool = True,
) -> Union[_HessianMatrix, _BatchHessianMatrix]:
    """
    Hessian (or batch Hessian) matrix from Hessian (or batch Hessian) dict.

    The ordering of the Hessian matrix blocks follows the ordering of the keys
    in `hessian_dict`.

    Args:
        model: Network model.
        inputs: Inputs (or batch inputs) to the model.
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

    Raises:
        RuntimeError: If `model` has an unsupported output shape.
    """
    # Get model parameters dict
    params_dict = dict(model.named_parameters())

    # Get Hessian parameter names
    hessian_param_names = list(hessian_dict.keys())

    # Determine batch size
    if is_batch:
        param_name = hessian_param_names[0]
        batch_size = hessian_dict[param_name][param_name].shape[0]
    else:
        # Non-batch case -- uses fictitious batch size of 1 for code commonality
        batch_size = 1

    # Determine output shape
    output_shape = _get_non_batched_output_shape(model, inputs, is_batch)

    # Determine output size
    sum_output_shape = sum(output_shape)
    zero_dim_output = sum_output_shape == 0
    output_size = max(1, sum_output_shape)

    # TODO: Handle high-dimensional output shapes
    if len(output_shape) >= 2:
        raise RuntimeError("Unsupported high-dimensional output shape")

    # Determine Hessian matrix size
    hessian_size = sum(params_dict[param_name].numel() for param_name in hessian_param_names)

    # Allocate Hessian or (batch Hessian) matrix
    hessian_matrix = torch.zeros(batch_size, output_size, hessian_size, hessian_size)

    # Populate batch Hessian matrix
    for batch in range(batch_size):
        for output in range(output_size):
            row_offset = 0
            for row_param_name in hessian_param_names:
                row_param_size = params_dict[row_param_name].numel()
                col_offset = 0
                for col_param_name in hessian_param_names:
                    col_param_size = params_dict[col_param_name].numel()

                    # Skip off-diagonal blocks in diagonal-only mode
                    if diagonal_only and row_param_name != col_param_name:
                        continue

                    # Get current Hessian block
                    index_tuple = (batch,) if is_batch else ()
                    if not zero_dim_output:
                        index_tuple += (output,)
                    hessian_block = hessian_dict[row_param_name][col_param_name][index_tuple]

                    # Add block to batch Hessian matrix
                    hessian_block = hessian_block.view(row_param_size, col_param_size)
                    row_slice = slice(row_offset, row_offset + row_param_size)
                    col_slice = slice(col_offset, col_offset + col_param_size)
                    hessian_matrix[batch, output, row_slice, col_slice] = hessian_block

                    # Advance column offset for next iteration
                    col_offset += col_param_size

                # Advance row offset for next iteration
                row_offset += row_param_size

    # Postprocess to remove fictitious dimensions
    if not is_batch:
        hessian_matrix.squeeze_(0)
        if zero_dim_output:
            hessian_matrix.squeeze_(0)
    else:
        if zero_dim_output:
            hessian_matrix.squeeze_(1)

    hessian_matrix.shape_metadata = {
        "batch_size": batch_size,
        "output_shape": output_shape,
        "output_size": output_size,
        "zero_dim_output": zero_dim_output,
    }

    return hessian_matrix


@jaxtyped(typechecker=typechecker)
def model_hessian_matrix(
    model: nn.Module,
    inputs: Union[Inputs, BatchInputs],
    params: Params = None,
    diagonal_only: bool = False,
    is_batch: bool = True,
) -> Union[_HessianMatrix, _BatchHessianMatrix]:
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
        inputs=inputs,
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
        target: Target output (or batch target output) from the model.
        params: Specific model parameters to use.  Default value is `None`,
            which means use all model parameters which are not frozen.
        diagonal_only: Make diagonal blocks only.  Default value is `False`.
        is_batch: Batch inputs and batch target provided.  Default value is
            `True`.

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
        inputs=inputs,
        hessian_dict=hessian_dict,
        diagonal_only=diagonal_only,
        is_batch=is_batch,
    )
