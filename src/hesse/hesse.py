"""
Hessian-related operations.

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

from typing import Any, Iterable, Optional, Union

from jaxtyping import Num, jaxtyped
from torch import Tensor, nn, vmap
from torch.func import functional_call, hessian
from typeguard import typechecked as typechecker

from src.hesse.types import Criterion

_HessianDict = dict[str, dict[str, Num[Tensor, "..."]]]

_ParamDict = dict[str, nn.Parameter]
_TensorDict = dict[str, Tensor]

_Params = Optional[Iterable[str]]
_Inputs = Union[Tensor, tuple[Tensor, ...]]
_BatchInputs = Union[Num[Tensor, "b ..."], tuple[Num[Tensor, "b ..."], ...]]
_Target = Union[Tensor, tuple[Tensor, ...]]
_BatchTarget = Num[Tensor, "b ..."]


def select_hessian_params(model: nn.Module, params: _Params = None) -> _ParamDict:
    """
    Select Hessian parameters to use.

    Frozen parameters (with `requires_grad = False`) are excluded.

    Args:
        model: Network model.
        params: Specific model parameters to use.

    Returns:
        Dict containing Hessian parameters.
    """
    hessian_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            if params is None or name in params:
                hessian_params[name] = param
    return hessian_params


def make_tuple(obj: Any) -> tuple:
    """
    Make object into tuple.

    Args:
        obj: Input object.

    Returns:
        This function returns `obj` if `obj` is a tuple.  Otherwise, it
        returns the 1-tuple containing `obj`.
    """
    return obj if isinstance(obj, tuple) else (obj,)


@jaxtyped(typechecker=typechecker)
def model_hessian(model: nn.Module, inputs: _Inputs, params: _Params = None) -> _HessianDict:
    """
    Hessian of a model with respect to its parameters.

    This function expects `inputs` to have no batch dimension.

    Args:
        model: Network model.
        inputs: Inputs to the model.
        params: Specific model parameters to use.  The default value is `None`
            which means use all model parameters which are not frozen.

    Returns:
        Hessian of `model` with respect to its parameters.

        The output `hess` is such that `hess["A"]["B"]` represents the Hessian
        matrix block corresponding to named parameters `A` and `B`.

    Raises:
        ValueError: If any arguments are invalid.
    """
    # Ensure `inputs` is always a tuple
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
    if not hessian_params:
        raise ValueError("No Hessian parameters selected")

    return hessian(functional_forward)(hessian_params)


@jaxtyped(typechecker=typechecker)
def batch_model_hessian(
    model: nn.Module, batch_inputs: _BatchInputs, params: _Params = None
) -> _HessianDict:
    """
    Batch Hessian of a model with respect to its parameters.

    Args:
        model: Network model.
        batch_inputs: Batch model inputs.
        params: Specific model parameters to use.  The default value is `None`
            which means use all model parameters which are not frozen.

    Returns:
        Batch Hessian of `model` with respect to its parameters.

        The output `hess` is such that `hess["A"]["B"][b, :]` represents the
        Hessian matrix block corresponding to batch `b` and named parameters
        `A` and `B`.
    """
    # Ensure `batch_inputs` is always a tuple
    batch_inputs = make_tuple(batch_inputs)

    # Note: Basic `Tensor` type hint is used here since `jaxtyping` is not
    # compatible with the `BatchedTensor` type produced by `vmap()`.
    def model_hessian_wrapper(inputs: Tensor) -> _HessianDict:
        """
        Wrap `model_hessian()` for vectorization with `torch.vmap()`.

        Args:
            inputs: Model inputs.

        Returns:
            Hessian result for the specified model inputs.
        """
        return model_hessian(model, inputs, params)

    return vmap(model_hessian_wrapper)(batch_inputs)


@jaxtyped(typechecker=typechecker)
def loss_hessian(
    model: nn.Module,
    criterion: Criterion,
    inputs: _Inputs,
    target: _Target,
    params: _Params = None,
) -> _HessianDict:
    """
    Hessian of a loss function with respect to model parameters.

    This version expects `inputs` and `targets` to have no batch dimension.

    Args:
        model: Network model.
        criterion: Loss criterion.
        inputs: Model inputs.
        target: Target model output.
        params: Specific model parameters to use.  The default value is `None`
            which means use all model parameters which are not frozen.

    Returns:
        Hessian of the loss function

            `loss = criterion(model(inputs), target)`

        with respect to model parameters.

        The output `hess` is such that `hess["A"]["B"]` represents the Hessian
        matrix block corresponding to named parameters `A` and `B`.

    Raises:
        ValueError: If any arguments are invalid.
    """
    # Ensure `inputs` is always a tuple
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
    if not hessian_params:
        raise ValueError("No Hessian parameters selected")

    return hessian(functional_loss)(hessian_params)


@jaxtyped(typechecker=typechecker)
def batch_loss_hessian(
    model: nn.Module,
    criterion: Criterion,
    batch_inputs: _BatchInputs,
    batch_target: _BatchTarget,
    params: _Params = None,
) -> _HessianDict:
    """
    Batch Hessian of a loss function with respect to model parameters.

    Args:
        model: Network model.
        criterion: Loss criterion.
        batch_inputs: Batch model inputs.
        batch_target: Batch target model output.
        params: Specific model parameters to use.

    Returns:
        Batch Hessian of the loss function

            `loss = criterion(model(inputs), target)`

        with respect to the specified model parameters.

        The output `hess` is such that `hess["A"]["B"][b, :]` represents the
        Hessian matrix block corresponding to batch `b` and named parameters
        `A` and `B`.
    """
    # Ensure `inputs` is always a tuple
    batch_inputs = make_tuple(batch_inputs)

    # Note: Basic `Tensor` type hint is used here since `jaxtyping` is not
    # compatible with the `BatchedTensor` type produced by `vmap()`.
    def loss_hessian_wrapper(inputs: Tensor, target: Tensor) -> _HessianDict:
        """
        Wrap `loss_hessian()` for vectorization with `torch.vmap()`.

        Args:
            inputs: Model inputs.
            target: Target model output.

        Returns:
            Hessian result for the specified model inputs and target.
        """
        return loss_hessian(model, criterion, inputs, target, params)

    return vmap(loss_hessian_wrapper)(batch_inputs, batch_target)
