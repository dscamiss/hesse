"""Utility functions."""

import torch
from jaxtyping import Num, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from src.hesse.types import HessianDict


@jaxtyped(typechecker=typechecker)
def make_hessian_matrix(model: nn.Module, hessian: HessianDict) -> Num[Tensor, "m n"]:
    """
    Make Hessian matrix from model and pre-computed Hessian data.

    The ordering of the matrix blocks follows the ordering of the keys in
    `hessian`.  We do not require the matrix to have a particular block
    ordering, since our only interest is its largest eigenvalue.

    Args:
        model: Model.
        hessian: Hessian data.

    Returns:
        Hessian matrix built up from the Hessian data.
    """
    # Get model parameters dict
    params_dict = dict(model.named_parameters())

    # Determine Hessian matrix size
    hessian_param_names = list(hessian.keys())
    hessian_size = sum(params_dict[param_name].numel() for param_name in hessian_param_names)

    # Allocate Hessian matrix
    hessian_matrix = torch.zeros(hessian_size, hessian_size)

    # Populate upper-triangular portion
    row_start = 0
    for row_param_name in hessian_param_names:
        row_size = params_dict[row_param_name].numel()
        row_end = row_start + row_size
        col_start = 0
        col_end = 0
        for col_param_name in hessian_param_names:
            col_size = params_dict[col_param_name].numel()
            col_end = col_start + col_size
            hessian_block = hessian[row_param_name][col_param_name].view(row_size, col_size)
            hessian_matrix[row_start:row_end, col_start:col_end] = hessian_block
            col_start = col_end
        row_start = row_end

    return hessian_matrix
