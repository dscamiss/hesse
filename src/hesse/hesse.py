"""Hessian-related operations."""

from jaxtyping import Num, jaxtyped
from torch import Tensor, nn
from torch.func import functional_call, hessian
from typeguard import typechecked as typechecker

from src.hesse.types import Hessian

_TensorTuple = tuple[Num[Tensor, "..."], ...]

# If the model has any buffers, please use make_functional_with_buffers() instead.


@jaxtyped(typechecker=typechecker)
def compute_hessian_old(model: nn.Module, *inputs: Num[Tensor, "..."]) -> Hessian:
    """
    Compute the Hessian of a model with respect to its parameters.

    Args:
        model: Network model.
        inputs: Inputs to the model.

    Returns:
        Hessian of `model` with respect to its parameters.
    """

    @jaxtyped(typechecker=typechecker)
    def wrap_model(params) -> Num[Tensor, "..."]:
        """
        Wrap a model to make it a function of its parameters.

        Args:
            params: Dict containing model parameters.

        Returns:
            Output of `model` with parameters `params`, evaluated on `inputs`.
        """
        nonlocal model, inputs
        return functional_call(model, params, inputs)

    params = dict(model.named_parameters())
    return hessian(wrap_model)(params)


@jaxtyped(typechecker=typechecker)
def compute_hessian(model: nn.Module, *inputs: Num[Tensor, "..."]) -> Hessian:
    """
    Compute the Hessian of a model with respect to its parameters.

    Args:
        model: Network model.
        inputs: Inputs to the model.

    Returns:
        Hessian of `model` with respect to its parameters.
    """

    @jaxtyped(typechecker=typechecker)
    def functional_forward(trainable_params) -> Num[Tensor, "..."]:
        """
        Wrap a model to make it a function of its parameters.

        Args:
            trainable_params: Dict containing trainable model parameters.

        Returns:
            Output of `model` with parameters `params`, evaluated on `inputs`.
        """
        return functional_call(model, trainable_params, inputs)

    trainable_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params[name] = param

    return hessian(functional_forward)(trainable_params)
