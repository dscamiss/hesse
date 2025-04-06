"""Test code for batch `loss_hessian_dict()`."""

# pylint: disable=invalid-name,too-many-statements

from collections import defaultdict

import pytest
import torch
from torch import nn

from src.hesse import loss_hessian_dict
from src.hesse.types import Criterion
from tests.conftest import commutation_matrix, randint


def test_batch_loss_hessian_dict_bilinear(
    bilinear: nn.Module, mse: Criterion, batch_size: int
) -> None:
    """Test with bilinear model."""
    # Make input data
    m, n = bilinear.B.in1_features, bilinear.B.in2_features
    x1 = randint((batch_size, m))
    x2 = randint((batch_size, n))
    batch_inputs = (x1, x2)
    batch_target = randint((batch_size, 1))  # Model output is (batch_size, 1)

    # Compute Hessian dict
    # - PyTorch issues performance warning for unimplemented batching rule
    # - This does not affect the correctness of the implementation.
    with pytest.warns(UserWarning):
        hessian_dict = loss_hessian_dict(
            model=bilinear,
            criterion=mse,
            inputs=batch_inputs,
            target=batch_target,
        )

    # Check keys
    err_str = "Key error"
    assert list(hessian_dict.keys()) == ["B.weight"], err_str
    assert list(hessian_dict["B.weight"].keys()) == ["B.weight"], err_str

    # Check Hessian shapes
    err_str = "Error in Hessian shapes"
    expected_shape = 2 * bilinear.B.weight.shape
    assert hessian_dict["B.weight"]["B.weight"].shape == expected_shape, err_str

    # Check Hessian values
    # - See comments in `test_loss_hessian_dict.py` for derivations.
    err_str = "Error in Hessian values"
    K_mn = commutation_matrix(m, n)

    # Accumulate expected Hessian values
    expected_value = torch.zeros(m * n, m * n)

    for batch in range(batch_size):
        flat_1 = torch.outer(x1[batch, :], x2[batch, :]).flatten()
        flat_2 = torch.outer(x2[batch, :], x1[batch, :]).flatten()
        expected_value += 2.0 * (flat_1.unsqueeze(-1) @ flat_2.unsqueeze(-1).T @ K_mn.T)

    actual_value = hessian_dict["B.weight"]["B.weight"].view(m * n, m * n)
    assert actual_value.equal(expected_value)


@pytest.mark.parametrize("diagonal_only", [True, False])
def test_batch_loss_hessian_dict_double_bilinear(
    double_bilinear: nn.Module,
    mse: Criterion,
    batch_size: int,
    diagonal_only: bool,
) -> None:
    """Test with double-bilinear model."""
    # Make aliases for brevity
    B1 = double_bilinear.B1
    B2 = double_bilinear.B2
    m, n, p = B1.shape[0], B1.shape[1], B2.shape[1]

    # Make input data
    x1 = randint((batch_size, m))
    x2 = randint((batch_size, p))
    batch_inputs = (x1, x2)
    batch_target = randint((batch_size,))  # Model output is (batch_size)

    # Compute Hessian dict
    hessian_dict = loss_hessian_dict(
        model=double_bilinear,
        criterion=mse,
        inputs=batch_inputs,
        target=batch_target,
        diagonal_only=diagonal_only,
    )

    # Check keys
    err_str = "Key error"
    assert list(hessian_dict.keys()) == ["B1", "B2"], err_str
    if not diagonal_only:
        assert list(hessian_dict["B1"].keys()) == ["B1", "B2"], err_str
        assert list(hessian_dict["B2"].keys()) == ["B1", "B2"], err_str
    else:
        assert list(hessian_dict["B1"].keys()) == ["B1"], err_str
        assert list(hessian_dict["B2"].keys()) == ["B2"], err_str

    # Check Hessian shapes
    err_str = "Error in Hessian shape"

    # Check (B1, B1) Hessian shape
    expected_shape = 2 * B1.shape
    assert hessian_dict["B1"]["B1"].shape == expected_shape, err_str

    # Check (B1, B2) Hessian shape
    if not diagonal_only:
        expected_shape = B1.shape + B2.shape
        assert hessian_dict["B1"]["B2"].shape == expected_shape, err_str

    # Check (B2, B1) Hessian shape
    if not diagonal_only:
        expected_shape = B2.shape + B1.shape
        assert hessian_dict["B2"]["B1"].shape == expected_shape, err_str

    # Check (B2, B2) Hessian shape
    expected_shape = 2 * B2.shape
    assert hessian_dict["B2"]["B2"].shape == expected_shape, err_str

    # Check Hessian values
    # - See comments in `test_loss_hessian_dict.py` for derivations.

    err_str = "Error in Hessian values"
    K_mn = commutation_matrix(m, n)
    K_np = commutation_matrix(n, p)

    # Accumulate expected Hessian values
    expected_value = defaultdict(dict)
    expected_value["B1"]["B1"] = torch.zeros(m * n, m * n)
    expected_value["B1"]["B2"] = torch.zeros(m * n, n * p)
    expected_value["B2"]["B1"] = torch.zeros(n * p, m * n)
    expected_value["B2"]["B2"] = torch.zeros(n * p, n * p)

    for batch in range(batch_size):
        outer_prod = torch.outer(x1[batch, :], x2[batch, :])
        err = double_bilinear(x1[batch, :], x2[batch, :]) - batch_target[batch]

        # Check (B1, B1) Hessian values
        prod = outer_prod @ B2.T
        prod_flat = prod.flatten().unsqueeze(-1)
        prod_transpose_flat = prod.T.flatten().unsqueeze(0)  # Implicit transpose
        expected_value["B1"]["B1"] += 2.0 * prod_flat @ prod_transpose_flat @ K_mn.T

        # Check (B1, B2) Hessian values
        if not diagonal_only:
            prod_left = outer_prod @ B2.T
            prod_right = outer_prod.T @ B1
            prod_left_flat = prod_left.flatten().unsqueeze(-1)
            prod_right_flat = prod_right.flatten().unsqueeze(0)  # Implicit transpose
            expected_value_1 = 2.0 * prod_left_flat @ prod_right_flat @ K_np.T
            expected_value_2 = 2.0 * err * K_mn @ torch.kron(torch.eye(n), outer_prod)
            expected_value["B1"]["B2"] += expected_value_1 + expected_value_2

        # Check (B2, B1) Hessian values
        if not diagonal_only:
            prod_left = B1.T @ outer_prod
            prod_right = B2 @ outer_prod.T
            prod_left_flat = prod_left.flatten().unsqueeze(-1)
            prod_right_flat = prod_right.flatten().unsqueeze(0)  # Implicit transpose
            expected_value_1 = 2.0 * prod_left_flat @ prod_right_flat @ K_mn.T
            # Recomputing the outer product is necessary here due to bugs:
            # - https://github.com/pytorch/pytorch/issues/54135
            # - https://github.com/pytorch/pytorch/issues/74442
            outer_prod_transpose = torch.outer(x2[batch, :], x1[batch, :])
            expected_value_2 = 2.0 * err * torch.kron(torch.eye(n), outer_prod_transpose) @ K_mn.T
            expected_value["B2"]["B1"] += expected_value_1 + expected_value_2

        # Check (B2, B2) Hessian values
        prod = B1.T @ outer_prod
        prod_flat = prod.flatten().unsqueeze(-1)
        prod_transpose_flat = prod.T.flatten().unsqueeze(0)  # Implicit transpose
        expected_value["B2"]["B2"] += 2.0 * prod_flat @ prod_transpose_flat @ K_np.T

    actual_value = hessian_dict["B1"]["B1"].view(m * n, m * n)
    assert actual_value.equal(expected_value["B1"]["B1"]), err_str

    if not diagonal_only:
        actual_value = hessian_dict["B1"]["B2"].view(m * n, n * p)
        assert actual_value.equal(expected_value["B1"]["B2"]), err_str

    if not diagonal_only:
        actual_value = hessian_dict["B2"]["B1"].view(n * p, m * n)
        assert actual_value.equal(expected_value["B2"]["B1"]), err_str

    actual_value = hessian_dict["B2"]["B2"].view(n * p, n * p)
    assert actual_value.equal(expected_value["B2"]["B2"]), err_str
