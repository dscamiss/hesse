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

from src.hesse.types import HessianDict

# Note: Here we use `Tensor` type hints here since `jaxtyping` is not
# compatible with the `BatchedTensor` type used by `vmap()`.


def compute_hessian(model: nn.Module, *inputs: Tensor) -> HessianDict:
    """
    Compute the Hessian of a model with respect to its parameters.

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
    def functional_forward(trainable_params) -> Num[Tensor, "..."]:
        """
        Wrap `model` to make it a function of its trainable parameters.

        Args:
            trainable_params: Dict containing trainable model parameters.

        Returns:
            The output of `model` with trainable parameters specified by
            `trainable_params`, evaluated at `inputs`.
        """
        return functional_call(model, trainable_params, inputs)

    trainable_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params[name] = param

    return hessian(functional_forward)(trainable_params)


@jaxtyped(typechecker=typechecker)
def compute_batch_hessian(model, *batch_inputs: Num[Tensor, "b ..."]) -> HessianDict:
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

    def compute_hessian_wrapper(inputs: Tensor) -> Tensor:
        """
        Wrap `compute_hessian()` for vectorization with `torch.vmap()`.

        Args:
            inputs: Inputs to the model.

        Returns:
            The output of `model` evaluated at `inputs`.
        """
        return compute_hessian(model, *inputs)

    return vmap(compute_hessian_wrapper)(batch_inputs)
