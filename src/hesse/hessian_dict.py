"""Functions to compute Hessian data."""

# Next line disables "returns Any" errors caused by unhinted PyTorch functions
# mypy: disable-error-code="no-any-return"

from collections import defaultdict
from typing import Union

from jaxtyping import Num, jaxtyped
from torch import Tensor, nn
from torch.func import functional_call, hessian
from typeguard import typechecked as typechecker

from hesse.types import (
    BatchHessianDict,
    BatchInputs,
    BatchTarget,
    Criterion,
    HessianDict,
    Inputs,
    Params,
    Target,
)
from hesse.utils import make_tuple

_ParamDict = dict[str, nn.Parameter]
_TensorDict = dict[str, Tensor]


def _select_hessian_params(model: nn.Module, params: Params = None) -> _ParamDict:
    """
    Select Hessian parameters to use.

    Frozen parameters (with `requires_grad = False`) are excluded.

    Args:
        model: Network model.
        params: Specific model parameters to use.  Default value is `None`,
            which means use all model parameters which are not frozen.

    Returns:
        Dict `hessian_params` that maps parameter names to parameters.  The
        keys of this dict are the selected parameter names.

    Raises:
        ValueError: If no Hessian parameters are selected.
    """
    hessian_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            if params is None or name in params:
                hessian_params[name] = param

    if not hessian_params:
        raise ValueError("No Hessian parameters selected")

    return hessian_params


@jaxtyped(typechecker=typechecker)
def _check_batch_dimension_across_inputs_and_hessians(
    inputs: tuple[Num[Tensor, "b ..."], ...],
    batch_hessian_dict: BatchHessianDict,
) -> bool:
    """
    Check for consistent batch dimension across inputs and Hessians.

    Args:
        inputs: Batch inputs to the model.
        batch_hessian_dict: Batch Hessian dict.

    Returns:
        `True` if and only if `inputs` and `batch_hessian_dict` have a
        consistent batch dimension.
    """
    ref_batch_dim = None

    # Check batch dimension across inputs
    for input_single in inputs:
        # Get reference batch dimension from first input
        if ref_batch_dim is None:
            ref_batch_dim = input_single.shape[0]
        else:
            if input_single.shape[0] != ref_batch_dim:
                return False

    # Check batch dimension across Hessians
    for param_name_outer in batch_hessian_dict.keys():
        for param_name_inner in batch_hessian_dict[param_name_outer].keys():
            hessian_single = batch_hessian_dict[param_name_outer][param_name_inner]
            if hessian_single.shape[0] != ref_batch_dim:
                return False

    return True


@jaxtyped(typechecker=typechecker)
def _check_num_dimensions_across_hessians(
    hessian_dict: Union[HessianDict, BatchHessianDict]
) -> bool:
    """
    Check for consistent number of dimensions across Hessians.

    Args:
        hessian_dict: Hessian (or batch Hessian) dict.

    Returns:
        Output is `True` if and only if `hessian_dict` has a consistent number
        of dimensions.
    """
    ref_ndim = None

    # Check number of dimensions across Hessians
    for param_name_outer in hessian_dict.keys():
        for param_name_inner in hessian_dict[param_name_outer].keys():
            hessian_single = hessian_dict[param_name_outer][param_name_inner]
            # Get reference number of dimensions from first Hessian
            if ref_ndim is None:
                ref_ndim = hessian_single.ndim
            else:
                if hessian_single.ndim != ref_ndim:
                    return False

    return True


@jaxtyped(typechecker=typechecker)
def model_hessian_dict(
    model: nn.Module,
    inputs: Union[Inputs, BatchInputs],
    params: Params = None,
    diagonal_only: bool = False,
    is_batch: bool = True,  # Only used in dimension check
) -> Union[HessianDict, BatchHessianDict]:
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
        Hessian (or batch Hessian) of `model` with respect to its parameters,
        represented as a dict.

        When `inputs` is batch, the output `hessian_dict` is such that
        `hessian_dict["P"]["Q"][b, :]` represents the Hessian matrix block
        corresponding to parameters named `P` and `Q` and batch `b`.

        When `inputs` is non-batch, the output `hessian_dict` is such that
        `hessian_dict["P"]["Q"]` represents the Hessian matrix block
        corresponding to parameters named `P` and `Q`.

        If `diagonal_only` is `True`, then the only valid keys for
        `hessian_dict` are of the form `hessian_dict["P"]["P"]`.

    Raises:
        RuntimeError: If the output fails various dimension checks.
    """
    # Ensure `inputs` is a tuple
    inputs = make_tuple(inputs)

    # Put model into eval mode to avoid side effects, ensure repeatability
    save_training_mode = model.training
    model.eval()

    # Type hint is `_TensorDict' here since `hessian()` changes data type
    @jaxtyped(typechecker=typechecker)
    def functional_forward(_params: _TensorDict) -> Num[Tensor, "..."]:
        """
        Wrap `model` to make it a function of specific model parameters.

        Args:
            _params: Specific model parameters to use.

        Returns:
            The value of `model(inputs)`, where `model` has parameters
            specified by `_params`.
        """
        return functional_call(model, _params, inputs)

    hessian_params = _select_hessian_params(model, params)

    if diagonal_only:
        # Handle `diagonal_only` case with multiple calls to `hessian()`
        hessian_dict: Union[HessianDict, BatchHessianDict] = defaultdict(dict)
        for param_name, param in hessian_params.items():
            hessian_single = hessian(functional_forward)({param_name: param})
            hessian_dict[param_name][param_name] = hessian_single[param_name][param_name]
    else:
        hessian_dict = hessian(functional_forward)(hessian_params)

    # Restore training mode
    model.train(save_training_mode)

    # Sanity check on number of dimensions
    if not _check_num_dimensions_across_hessians(hessian_dict):
        raise RuntimeError("Mismatched number of dimensions")

    # Sanity check on batch dimensions, if necessary
    if is_batch:
        if not _check_batch_dimension_across_inputs_and_hessians(inputs, hessian_dict):
            raise RuntimeError("Mismatched batch dimensions")

    return hessian_dict


@jaxtyped(typechecker=typechecker)
def loss_hessian_dict(
    model: nn.Module,
    criterion: Criterion,
    inputs: Union[Inputs, BatchInputs],
    target: Union[Target, BatchTarget],
    params: Params = None,
    diagonal_only: bool = False,
) -> HessianDict:
    """
    Hessian of loss function with respect to model parameters.

    Args:
        model: Network model.
        criterion: Loss criterion.
        inputs: Inputs (or batch inputs) to the model.
        target: Target output (or batch target output) from the model.
        params: Specific model parameters to use.  Default value is `None`,
            which means use all model parameters which are not frozen.
        diagonal_only: Make diagonal blocks only.  Default value is `False`.

    Returns:
        Hessian of the loss function `criterion(model(inputs), target)` with
        respect to the parameters of `model`, represented as a dict.

        The output `hessian_dict` is such that `hessian_dict["P"]["Q"]`
        represents the Hessian matrix block corresponding to parameters named
        `P` and `Q`.

        If `diagonal_only` is `True`, then the only valid keys for
        `hessian_dict` are of the form `hessian_dict["P"]["P"]`.

        Note that `hessian_dict` is non-batch, even if `inputs` is batch.

    Raises:
        RuntimeError: If the output fails various dimension checks.
    """
    # Ensure `inputs` is a tuple
    inputs = make_tuple(inputs)

    # Put model into eval mode to avoid side effects, ensure repeatability
    save_training_mode = model.training
    model.eval()

    # Type hint is `_TensorDict' here since `hessian()` changes data type
    @jaxtyped(typechecker=typechecker)
    def functional_loss(_params: _TensorDict) -> Num[Tensor, ""]:
        """
        Wrap `loss` to make it a function of specific model parameters.

        Args:
            _params: Specific model parameters to use.

        Returns:
            The value of `criterion(model(inputs), targets)`, where `model`
            has parameters specified by `_params`.
        """
        output = functional_call(model, _params, inputs)
        return criterion(output, target)

    hessian_params = _select_hessian_params(model, params)

    if diagonal_only:
        # Handle `diagonal_only` case with multiple calls to `hessian()`
        hessian_dict: HessianDict = defaultdict(dict)
        for param_name, param in hessian_params.items():
            hessian_single = hessian(functional_loss)({param_name: param})
            hessian_dict[param_name][param_name] = hessian_single[param_name][param_name]
    else:
        hessian_dict = hessian(functional_loss)(hessian_params)

    # Restore training mode
    model.train(save_training_mode)

    # Sanity check on number of dimensions
    if not _check_num_dimensions_across_hessians(hessian_dict):
        raise RuntimeError("Mismatched number of dimensions")

    # Sanity check on batch dimension (we expect no batch dimension)
    param_name = next(iter(hessian_dict.keys()))
    if hessian_dict[param_name][param_name].ndim == 3:
        raise RuntimeError("Unexpected batch dimension")

    return hessian_dict
