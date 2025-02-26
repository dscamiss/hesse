"""Test code for utils."""

# pylint: disable=invalid-name

from typing import Callable

import torch


@torch.no_grad()
def test_commutation_matrix(commutation_matrix: Callable) -> None:
    """Test `commutation_matrix()`."""
    A = torch.randn(3, 4)
    K = commutation_matrix(A.shape[0], A.shape[1])

    err_str = "Error in commutation matrix"
    # Note: Transpose here since `flatten()` uses row-major component ordering
    vec_A = A.T.flatten()
    vec_A_transpose = A.flatten()
    assert torch.all(K @ vec_A == vec_A_transpose), err_str
