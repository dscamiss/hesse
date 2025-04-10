# `hesse` 🧘‍♂️

![License](https://img.shields.io/badge/license-MIT-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Build](https://github.com/dscamiss/hesse/actions/workflows/python-package.yml/badge.svg)
[![codecov](https://codecov.io/gh/dscamiss/hesse/graph/badge.svg?token=Z3CGGZJ70B)](https://codecov.io/gh/dscamiss/hesse)

# Introduction

The goal of `hesse` is to simplify the computation of Hessian matrices (and related quantities) in PyTorch.  

In particular, the goal is to simplify the computation of:

* **Model Hessians** (these are Hessian matrices of a given model with respect to its trainable parameters);
* **Loss function Hessians** (these are Hessian matrices of a loss function of the form `criterion(model(inputs), target)` with respect to `model`'s trainable parameters).

This is achieved with user-friendly wrappers for `torch.func` transforms.

# Installation

In an existing Python 3.9+ environment:

```bash
git clone https://github.com/dscamiss/hesse/
pip install ./hesse
```

# Examples

## Setup

Create a toy multi-input, multi-output model.

```python
import torch
from torch import Tensor

class MimoModel(torch.nn.Module):
    """Multi-input, multi-output demo model."""

    def __init__(self, m: int) -> None:
        super().__init__()
        self.A = torch.nn.Parameter(torch.randn(m, m))
        self.B = torch.nn.Parameter(torch.randn(m, m))

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Num[Tensor, "b n"], y: Num[Tensor, "b n"]) -> Num[Tensor, "b two_n"]:
        """
        Run forward pass.

        Args:
            x: First input tensor of shape (b, n).
            y: Second input tensor of shape (b, n).

        Returns:
            The matrix

            [ tr(A^t A) x_{    0, :}   tr(B^t B) y_{    0, :} ]
            [ tr(A^t A) x_{    1, :}   tr(B^t B) y_{    1, :} ]
            [           :                        :            ]
            [ tr(A^t A) x_{m - 1, :}   tr(B^t B) y_{m - 1, :} ].
        """
        row_1 = torch.trace(self.A.T @ self.A) * x
        row_2 = torch.trace(self.B.T @ self.B) * y
        return torch.hstack((row_1, row_2))
```

Make an instance of `MimoModel` and batch inputs.

```python
model = MimoModel(2)

x = torch.Tensor(
    [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
)
y = -1.0 * x
```

## Model Hessians

Computing the Hessian matrix of `model` is easy:

```python
hessian = hesse.model_hessian_matrix(model=model, inputs=(x, y))
```

Generally speaking, the shape of the Hessian matrix will be `(batch_size, output_size, ...)`.  In this instance, `batch_size = 2` and `output_size = 4`, 
so that `hessian` has shape `(2, 4, 8, 8)`.

To compute the Hessian matrix with respect to a subset of the model parameters, just provide a list of parameter names:

```python
hessian = hesse.model_hessian_matrix(model=model, inputs=(x, y), params=["A"])
```

## Loss function Hessians

Create a loss criterion and batch target output:

```python
criterion = torch.nn.MSELoss()
target = torch.randn(2, 4)
```

Computing the Hessian matrix of the loss function `criterion(model(inputs), target)` is easy:

```python
loss_hessian = hesse.loss_hessian_matrix(
    model=model,
    criterion=criterion,
    inputs=(x, y),
    target=target,
)
```

As above, we can compute the Hessian matrix with respect to a subset of the model parameters:

```python
loss_hessian = hesse.loss_hessian_matrix(
    model=model,
    criterion=criterion,
    inputs=(x, y),
    target=target,
    params=["A"],
)
```

