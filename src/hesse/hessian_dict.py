"""
Helper functions to compute Hessian data.

The functions in this module are concerned with Hessians of models with
respect to their parameters.

For a model f : R^m --> R with parameters P_1, ..., P_r, the Hessian of f
with respect to its parameters, evaluated at the input x, is the k-by-k block
matrix

    Hess(f)(x) = [Hess_{i,j}(f)(x)],

where

                                     d^2 f(x)
    (Hess_{i,j}(f)(x))_{k,l} = ---------------------
                               d(P_i)_{k} d(P_j)_{l}

and each parameter P_i has the row-major component ordering.

To evaluate Hess(f)(x) as a bilinear form, we can use the identity

    vec(P_1, ..., P_k)^t Hess(f)(x) vec(Q_1, ..., Q_k)
        = sum_{i,j=1}^k vec(P_i)^t Hess_{i,j}(f)(x) vec(Q_j),

where vec() is the row-major vectorization map.
"""

# Next line disables "returns Any" errors caused by unhinted PyTorch functions
# mypy: disable-error-code="no-any-return"

from collections import defaultdict
from typing import Union

from jaxtyping import Num, jaxtyped
from torch import Tensor, nn
from torch.func import functional_call, hessian
from typeguard import typechecked as typechecker

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
from src.hesse.utils import make_tuple

_ParamDict = dict[str, nn.Parameter]
_TensorDict = dict[str, Tensor]


def select_hessian_params(model: nn.Module, params: Params = None) -> _ParamDict:
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
def model_hessian_dict(
    model: nn.Module, inputs: Inputs, params: Params = None, diagonal_only: bool = False
) -> Union[HessianDict, BatchHessianDict]:
    """
    Hessian (or batch Hessian) of a model with respect to its parameters.

    Args:
        model: Network model.
        inputs: Inputs (or batch inputs) to the model.
        params: Specific model parameters to use.  Default value is `None`,
            which means use all model parameters which are not frozen.
        diagonal_only: Make diagonal blocks only.  Default value is `False`.

    Returns:
        Hessian (or batch Hessian) of `model` with respect to its parameters,
        represented as a dict.

        When `inputs` is not batched, the output `hessian_dict` is such that
        `hessian_dict["P"]["Q"]` represents the Hessian matrix block
        corresponding to parameters named `P` and `Q`.

        When `inputs` is batched, the output `hessian_dict` is such that
        `hessian_dict["P"]["Q"][b, :]` represents the Hessian matrix block
        corresponding to parameters named `P` and `Q` and batch `b`.

        If `diagonal_only` is `True`, then the only valid keys for
        `hessian_dict` are of the form `hessian_dict["P"]["P"]`.
    """
    # Ensure `inputs` is a tuple
    inputs = make_tuple(inputs)

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

    hessian_params = select_hessian_params(model, params)

    if diagonal_only:
        # Handle `diagonal_only` case with multiple calls to `hessian()`
        hessian_dict: Union[HessianDict, BatchHessianDict] = defaultdict(dict)
        for param_name, param in hessian_params.items():
            hessian_single = hessian(functional_forward)({param_name: param})
            hessian_dict[param_name][param_name] = hessian_single[param_name][param_name]
    else:
        hessian_dict = hessian(functional_forward)(hessian_params)

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
    Hessian of a loss function with respect to model parameters.

    Args:
        model: Network model.
        criterion: Loss criterion.
        inputs: Inputs (or batch inputs) to the model.
        target: Target outputs (or batch outputs) from the model.
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

        Note that `hessian_dict` is not batched, even if `inputs` is batched.
        The loss function is computed using the entire batch.
    """
    # Ensure `inputs` is a tuple
    inputs = make_tuple(inputs)

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

    hessian_params = select_hessian_params(model, params)

    if diagonal_only:
        # Handle `diagonal_only` case with multiple calls to `hessian()`
        hessian_dict: HessianDict = defaultdict(dict)
        for param_name, param in hessian_params.items():
            hessian_single = hessian(functional_loss)({param_name: param})
            hessian_dict[param_name][param_name] = hessian_single[param_name][param_name]
    else:
        hessian_dict = hessian(functional_loss)(hessian_params)

    return hessian_dict
