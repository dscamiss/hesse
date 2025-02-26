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

from jaxtyping import Num, jaxtyped
from torch import Tensor, nn, vmap
from torch.func import functional_call, hessian
from typeguard import typechecked as typechecker

from src.hesse.types import HessianDict, Loss, ParamDict

# Note: Here we use `Tensor` type hints here since `jaxtyping` is not
# compatible with the `BatchedTensor` type used by `vmap()`.


def compute_model_hessian(model: nn.Module, *inputs: Tensor) -> HessianDict:
    """
    Compute the Hessian of a model with respect to its parameters.

    This function expects `inputs` to have no batch dimension.

    Args:
        model: Network model.
        inputs: Inputs to the model.

    Returns:
        Hessian of `model` with respect to its parameters.

        The output `hess` is such that `hess["A"]["B"]` represents the Hessian
        matrix block corresponding to named parameters `A` and `B`.

    Note:
        Frozen parameters are not included.
    """

    @jaxtyped(typechecker=typechecker)
    def functional_forward(params: ParamDict) -> Num[Tensor, "..."]:
        """
        Wrap `model` to make it a function of specific model parameters.

        Args:
            params: Specific model parameters to use.

        Returns:
            The output of `model` with parameters specified by `params`,
            evaluated at `inputs`.
        """
        return functional_call(model, params, inputs)

    trainable_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params[name] = param

    return hessian(functional_forward)(trainable_params)


def compute_loss_hessian(
    model: nn.Module, criterion: Loss, *inputs: Tensor, targets: Tensor
) -> HessianDict:
    """
    Compute the Hessian of a loss function with respect to model parameters.

    This version expects `inputs` and `targets` to have no batch dimension.

    Args:
        model: Network model.
        criterion: Loss criterion.
        inputs: Inputs to the model.
        targets: Target outputs from the model.

    Returns:
        Hessian of the loss function

            `loss = criterion(model(inputs), targets)`

        with respect to model parameters.

        The output `hess` is such that `hess["A"]["B"]` represents the Hessian
        matrix block corresponding to named parameters `A` and `B`.

    Note:
        Frozen parameters are not included.
    """

    @jaxtyped(typechecker=typechecker)
    def functional_forward(params: ParamDict) -> Num[Tensor, ""]:
        """
        Wrap `loss` to make it a function of specific model parameters.

        Args:
            params: Specific model parameters to use.

        Returns:
            The value of `criterion(model(inputs), targets)`, where `model`
            has parameters specified by `params`.
        """
        outputs = functional_call(model, params, inputs)
        return criterion(outputs, targets)

    trainable_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params[name] = param

    return hessian(functional_forward)(trainable_params)


@jaxtyped(typechecker=typechecker)
def compute_batch_model_hessian(model, *batch_inputs: Num[Tensor, "b ..."]) -> HessianDict:
    """
    Compute the batch Hessian of a model with respect to its parameters.

    Args:
        model: Network model.
        batch_inputs: Batch inputs to the model.

    Returns:
        Batch Hessian of `model` with respect to its parameters.

        The output `hess` is such that `hess["A"]["B"][b, :]` represents the
        Hessian matrix block corresponding to batch `b` and named parameters
        `A` and `B`.

    Note:
        Frozen parameters are not included.
    """

    def compute_model_hessian_wrapper(inputs: Tensor) -> Tensor:
        """
        Wrap `compute_model_hessian()` for vectorization with `torch.vmap()`.

        Args:
            inputs: Inputs to the model.

        Returns:
            The output of `model` evaluated at `inputs`.
        """
        return compute_model_hessian(model, *inputs)

    return vmap(compute_model_hessian_wrapper)(batch_inputs)
