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

from jaxtyping import Num, jaxtyped
from torch import Tensor, nn, vmap
from torch.func import functional_call, hessian
from typeguard import typechecked as typechecker

from src.hesse.types import BatchInputs, BatchTarget, Criterion, HessianDict, Inputs, Params, Target
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
        Dict containing Hessian parameters.
    """
    hessian_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            if params is None or name in params:
                hessian_params[name] = param
    return hessian_params


@jaxtyped(typechecker=typechecker)
def model_hessian_dict(
    model: nn.Module, inputs: Inputs, params: Params = None, diagonal_only: bool = False
) -> HessianDict:
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
        Hessian of `model` with respect to its parameters, represented as a
        dict.

        The output `hess` is such that `hess["A"]["B"]` represents the Hessian
        matrix block corresponding to named parameters `A` and `B`.

    Raises:
        ValueError: If any arguments are invalid.
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
    if not hessian_params:
        raise ValueError("No Hessian parameters selected")

    if diagonal_only:
        # Handle `diagonal_only` case with multiple calls to `hessian()`
        hess: HessianDict = defaultdict(dict)
        for param_name, param in hessian_params.items():
            hess_single = hessian(functional_forward)({param_name: param})
            hess[param_name][param_name] = hess_single[param_name][param_name]
    else:
        hess = hessian(functional_forward)(hessian_params)
    return hess


@jaxtyped(typechecker=typechecker)
def batch_model_hessian_dict(
    model: nn.Module, batch_inputs: BatchInputs, params: Params = None, diagonal_only: bool = False
) -> HessianDict:
    """
    Batch Hessian of a model with respect to its parameters.

    Args:
        model: Network model.
        batch_inputs: Batch model inputs.
        params: Specific model parameters to use.  Default value is `None`,
            which means use all model parameters which are not frozen.
        diagonal_only: Make diagonal data only.  Default value is `False`.

    Returns:
        Batch Hessian of `model` with respect to its parameters, represented
        as a dict.

        The output `hess` is such that `hess["A"]["B"][b, :]` represents the
        Hessian matrix block corresponding to batch `b` and named parameters
        `A` and `B`.
    """
    # Ensure `batch_inputs` is a tuple
    batch_inputs = make_tuple(batch_inputs)

    # Note: Basic `Tensor` type hint is used here since `jaxtyping` is not
    # compatible with the `BatchedTensor` type produced by `vmap()`.
    def model_hessian_dict_wrapper(inputs: Tensor) -> HessianDict:
        """
        Wrap `model_hessian_dict()` for vectorization with `torch.vmap()`.

        Args:
            inputs: Model inputs.

        Returns:
            Hessian result for the specified model inputs.
        """
        return model_hessian_dict(model, inputs, params, diagonal_only)

    return vmap(model_hessian_dict_wrapper)(batch_inputs)


@jaxtyped(typechecker=typechecker)
def loss_hessian_dict(
    model: nn.Module,
    criterion: Criterion,
    inputs: Inputs,
    target: Target,
    params: Params = None,
    diagonal_only: bool = False,
) -> HessianDict:
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
        Hessian of the loss function

            `loss = criterion(model(inputs), target)`

        with respect to model parameters, represented as a dict.

        The output `hess` is such that `hess["A"]["B"]` represents the Hessian
        matrix block corresponding to named parameters `A` and `B`.

    Raises:
        ValueError: If any arguments are invalid.
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
    if not hessian_params:
        raise ValueError("No Hessian parameters selected")

    if diagonal_only:
        # Handle `diagonal_only` case with multiple calls to `hessian()`
        hess: HessianDict = defaultdict(dict)
        for param_name, param in hessian_params.items():
            hess_single = hessian(functional_loss)({param_name: param})
            hess[param_name][param_name] = hess_single[param_name][param_name]
    else:
        hess = hessian(functional_loss)(hessian_params)
    return hess


@jaxtyped(typechecker=typechecker)
def batch_loss_hessian_dict(
    model: nn.Module,
    criterion: Criterion,
    batch_inputs: BatchInputs,
    batch_target: BatchTarget,
    params: Params = None,
    diagonal_only: bool = False,
) -> HessianDict:
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

        with respect to the specified model parameters, represented as a dict.

        The output `hess` is such that `hess["A"]["B"][b, :]` represents the
        Hessian matrix block corresponding to batch `b` and named parameters
        `A` and `B`.
    """
    # Ensure `batch_inputs` is a tuple
    batch_inputs = make_tuple(batch_inputs)

    # Note: Basic `Tensor` type hint is used here since `jaxtyping` is not
    # compatible with the `BatchedTensor` type produced by `vmap()`.
    def loss_hessian_dict_wrapper(inputs: Tensor, target: Tensor) -> HessianDict:
        """
        Wrap `loss_hessian_dict()` for vectorization with `torch.vmap()`.

        Args:
            inputs: Model inputs.
            target: Target model output.

        Returns:
            Hessian result for the specified model inputs and target.
        """
        return loss_hessian_dict(model, criterion, inputs, target, params, diagonal_only)

    return vmap(loss_hessian_dict_wrapper)(batch_inputs, batch_target)
